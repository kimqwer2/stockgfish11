#include "transform.h"

#include "sfen_stream.h"
#include "packed_sfen.h"
#include "sfen_writer.h"

#include "thread.h"
#include "position.h"
#include "evaluate.h"
#include "search.h"

#include "nnue/evaluate_nnue.h"

#include <string>
#include <map>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <fstream>
#include <cstdio>

namespace Stockfish::Tools
{
    using CommandFunc = void(*)(std::istringstream&);

    enum struct NudgedStaticMode
    {
        Absolute,
        Relative,
        Interpolate
    };

    struct NudgedStaticParams
    {
        std::string input_filename = "in.bin";
        std::string output_filename = "out.bin";
        NudgedStaticMode mode = NudgedStaticMode::Absolute;
        int absolute_nudge = 5;
        float relative_nudge = 0.1;
        float interpolate_nudge = 0.1;

        void enforce_constraints()
        {
            relative_nudge = std::max(relative_nudge, 0.0f);
            absolute_nudge = std::max(absolute_nudge, 0);
        }
    };


    struct MineParams
    {
        std::string mode = "search-gap";
        std::string input_filename = "in.bin";
        std::string output_filename = "hard.bin";
        std::string net;
        std::string old_net;
        std::string new_net;
        std::uint64_t keep_count = 1000000;
        std::uint64_t batch_size = 100000;
    };

    struct MinedEntry
    {
        int gap;
        std::uint64_t ordinal;
        PackedSfenValue psv;

        bool operator>(const MinedEntry& other) const
        {
            if (gap != other.gap)
                return gap > other.gap;
            return ordinal > other.ordinal;
        }
    };

    static bool load_nnue_file(const std::string& filename)
    {
        std::ifstream stream(filename, std::ios::binary);
        if (!stream)
        {
            std::cerr << "ERROR: Could not open NNUE file " << filename << '\n';
            return false;
        }
        currentNnueVariant = variants.find(Options["UCI_Variant"])->second;
        if (!Eval::NNUE::load_eval(filename, stream))
        {
            std::cerr << "ERROR: Could not load NNUE file " << filename << '\n';
            return false;
        }
        return true;
    }

    static void add_mined_entry(std::priority_queue<MinedEntry, std::vector<MinedEntry>, std::greater<MinedEntry>>& heap,
                                std::uint64_t keep_count,
                                int gap,
                                std::uint64_t ordinal,
                                const PackedSfenValue& psv)
    {
        if (keep_count == 0)
            return;

        MinedEntry entry{gap, ordinal, psv};
        if (heap.size() < keep_count)
            heap.push(entry);
        else if (heap.top().gap < gap || (heap.top().gap == gap && heap.top().ordinal > ordinal))
        {
            heap.pop();
            heap.push(entry);
        }
    }

    static void write_mined_entries(std::priority_queue<MinedEntry, std::vector<MinedEntry>, std::greater<MinedEntry>>& heap,
                                    const std::string& output_filename,
                                    std::uint64_t batch_size)
    {
        std::vector<MinedEntry> entries;
        entries.reserve(heap.size());
        while (!heap.empty())
        {
            entries.emplace_back(heap.top());
            heap.pop();
        }

        std::sort(entries.begin(), entries.end(), [](const MinedEntry& a, const MinedEntry& b) {
            if (a.gap != b.gap)
                return a.gap > b.gap;
            return a.ordinal < b.ordinal;
        });

        std::remove(Tools::filename_with_extension(output_filename, Tools::BinSfenOutputStream::extension).c_str());
        auto out = Tools::create_new_sfen_output(output_filename);
        if (out == nullptr)
        {
            std::cerr << "Invalid output file type.\n";
            return;
        }

        PSVector buffer;
        buffer.reserve(batch_size);
        for (const auto& entry : entries)
        {
            buffer.emplace_back(entry.psv);
            if (buffer.size() >= batch_size)
            {
                out->write(buffer);
                buffer.clear();
            }
        }
        if (!buffer.empty())
            out->write(buffer);

        std::cout << "Wrote " << entries.size() << " mined positions to " << output_filename << "\n";
    }

    static void do_mine_search_gap(MineParams& params)
    {
        if (!load_nnue_file(params.net))
            return;

        Thread* th = Threads.main();
        Position& pos = th->rootPos;
        StateInfo si;
        auto in = Tools::open_sfen_input_file(params.input_filename);
        if (in == nullptr)
        {
            std::cerr << "Invalid input file type.\n";
            return;
        }

        std::priority_queue<MinedEntry, std::vector<MinedEntry>, std::greater<MinedEntry>> heap;
        std::uint64_t processed = 0;
        for (;;)
        {
            auto v = in->next();
            if (!v.has_value())
                break;
            auto psv = v.value();
            pos.set_from_packed_sfen(psv.sfen, &si, th);
            int gap = std::abs(int(psv.score) - int(Eval::NNUE::evaluate(pos)));
            add_mined_entry(heap, params.keep_count, gap, processed, psv);
            if (++processed % 1000000 == 0)
                std::cout << "Processed " << processed << " positions. Current cutoff gap " << (heap.empty() ? 0 : heap.top().gap) << "\n";
        }
        write_mined_entries(heap, params.output_filename, params.batch_size);
    }

    static void do_mine_eval_disagree(MineParams& params)
    {
        const std::string temp_filename = params.output_filename + ".old_eval.tmp";
        {
            if (!load_nnue_file(params.old_net))
                return;
            Thread* th = Threads.main();
            Position& pos = th->rootPos;
            StateInfo si;
            auto in = Tools::open_sfen_input_file(params.input_filename);
            std::ofstream temp(temp_filename, std::ios::binary | std::ios::trunc);
            if (in == nullptr || !temp)
            {
                std::cerr << "Invalid input file type or temporary file.\n";
                return;
            }
            std::uint64_t processed = 0;
            for (;;)
            {
                auto v = in->next();
                if (!v.has_value())
                    break;
                pos.set_from_packed_sfen(v->sfen, &si, th);
                std::int16_t eval = static_cast<std::int16_t>(std::clamp(int(Eval::NNUE::evaluate(pos)), int(std::numeric_limits<std::int16_t>::min()), int(std::numeric_limits<std::int16_t>::max())));
                temp.write(reinterpret_cast<const char*>(&eval), sizeof(eval));
                if (++processed % 1000000 == 0)
                    std::cout << "Old net pass processed " << processed << " positions.\n";
            }
        }

        if (!load_nnue_file(params.new_net))
            return;
        Thread* th = Threads.main();
        Position& pos = th->rootPos;
        StateInfo si;
        auto in = Tools::open_sfen_input_file(params.input_filename);
        std::ifstream temp(temp_filename, std::ios::binary);
        if (in == nullptr || !temp)
        {
            std::cerr << "Invalid input file type or temporary file.\n";
            std::remove(temp_filename.c_str());
            return;
        }

        std::priority_queue<MinedEntry, std::vector<MinedEntry>, std::greater<MinedEntry>> heap;
        std::uint64_t processed = 0;
        for (;;)
        {
            auto v = in->next();
            std::int16_t old_eval;
            if (!v.has_value() || !temp.read(reinterpret_cast<char*>(&old_eval), sizeof(old_eval)))
                break;
            auto psv = v.value();
            pos.set_from_packed_sfen(psv.sfen, &si, th);
            int gap = std::abs(int(old_eval) - int(Eval::NNUE::evaluate(pos)));
            add_mined_entry(heap, params.keep_count, gap, processed, psv);
            if (++processed % 1000000 == 0)
                std::cout << "New net pass processed " << processed << " positions. Current cutoff gap " << (heap.empty() ? 0 : heap.top().gap) << "\n";
        }
        std::remove(temp_filename.c_str());
        write_mined_entries(heap, params.output_filename, params.batch_size);
    }

    void mine(std::istringstream& is)
    {
        MineParams params;
        for (;;)
        {
            std::string token;
            is >> token;
            if (token.empty()) break;
            if (token == "mode" || token == "--mode") is >> params.mode;
            else if (token == "input_file" || token == "--input") is >> params.input_filename;
            else if (token == "output_file" || token == "--output") is >> params.output_filename;
            else if (token == "net" || token == "--net") is >> params.net;
            else if (token == "old_net" || token == "--old-net") is >> params.old_net;
            else if (token == "new_net" || token == "--new-net") is >> params.new_net;
            else if (token == "keep_count" || token == "--keep-count") is >> params.keep_count;
            else if (token == "batch_size" || token == "--batch-size") is >> params.batch_size;
            else { std::cerr << "ERROR: Unknown option " << token << "\n"; return; }
        }
        params.batch_size = std::max<std::uint64_t>(1, params.batch_size);
        std::cout << "Hard position mining mode=" << params.mode << " input=" << params.input_filename << " output=" << params.output_filename << " keep_count=" << params.keep_count << "\n";
        if (params.mode == "search-gap")
        {
            if (params.net.empty()) { std::cerr << "ERROR: search-gap requires --net.\n"; return; }
            do_mine_search_gap(params);
        }
        else if (params.mode == "eval-disagree")
        {
            if (params.old_net.empty() || params.new_net.empty()) { std::cerr << "ERROR: eval-disagree requires --old-net and --new-net.\n"; return; }
            do_mine_eval_disagree(params);
        }
        else std::cerr << "ERROR: Unsupported mining mode " << params.mode << "\n";
    }

    struct RescoreParams
    {
        std::string input_filename = "in.epd";
        std::string output_filename = "out.bin";
        int depth = 3;
        int research_count = 0;
        bool keep_moves = true;

        void enforce_constraints()
        {
            depth = std::max(1, depth);
            research_count = std::max(0, research_count);
        }
    };

    [[nodiscard]] std::int16_t nudge(NudgedStaticParams& params, std::int16_t static_eval_i16, std::int16_t deep_eval_i16)
    {
        auto saturate_i32_to_i16 = [](int v) {
            return static_cast<std::int16_t>(
                std::clamp(
                    v,
                    (int)std::numeric_limits<std::int16_t>::min(),
                    (int)std::numeric_limits<std::int16_t>::max()
                )
            );
        };

        auto saturate_f32_to_i16 = [saturate_i32_to_i16](float v) {
            return saturate_i32_to_i16((int)v);
        };

        int static_eval = static_eval_i16;
        int deep_eval = deep_eval_i16;

        switch(params.mode)
        {
            case NudgedStaticMode::Absolute:
                return saturate_i32_to_i16(
                    static_eval + std::clamp(
                        deep_eval - static_eval,
                        -params.absolute_nudge,
                        params.absolute_nudge
                    )
                );

            case NudgedStaticMode::Relative:
                return saturate_f32_to_i16(
                    (float)static_eval * std::clamp(
                        (float)deep_eval / (float)static_eval,
                        (1.0f - params.relative_nudge),
                        (1.0f + params.relative_nudge)
                    )
                );

            case NudgedStaticMode::Interpolate:
                return saturate_f32_to_i16(
                    (float)static_eval * (1.0f - params.interpolate_nudge)
                    + (float)deep_eval * params.interpolate_nudge
                );

            default:
                assert(false);
                return 0;
        }
    }

    void do_nudged_static(NudgedStaticParams& params)
    {
        Thread* th = Threads.main();
        Position& pos = th->rootPos;
        StateInfo si;

        auto in = Tools::open_sfen_input_file(params.input_filename);
        auto out = Tools::create_new_sfen_output(params.output_filename);

        if (in == nullptr)
        {
            std::cerr << "Invalid input file type.\n";
            return;
        }

        if (out == nullptr)
        {
            std::cerr << "Invalid output file type.\n";
            return;
        }

        PSVector buffer;
        uint64_t batch_size = 1'000'000;

        buffer.reserve(batch_size);

        uint64_t num_processed = 0;
        for (;;)
        {
            auto v = in->next();
            if (!v.has_value())
                break;

            auto& ps = v.value();

            pos.set_from_packed_sfen(ps.sfen, &si, th);
            auto static_eval = Eval::evaluate(pos);
            auto deep_eval = ps.score;
            ps.score = nudge(params, static_eval, deep_eval);

            buffer.emplace_back(ps);
            if (buffer.size() >= batch_size)
            {
                num_processed += buffer.size();

                out->write(buffer);
                buffer.clear();

                std::cout << "Processed " << num_processed << " positions.\n";
            }
        }

        if (!buffer.empty())
        {
            num_processed += buffer.size();

            out->write(buffer);
            buffer.clear();

            std::cout << "Processed " << num_processed << " positions.\n";
        }

        std::cout << "Finished.\n";
    }

    void nudged_static(std::istringstream& is)
    {
        NudgedStaticParams params{};

        while(true)
        {
            std::string token;
            is >> token;

            if (token == "")
                break;

            if (token == "absolute")
            {
                params.mode = NudgedStaticMode::Absolute;
                is >> params.absolute_nudge;
            }
            else if (token == "relative")
            {
                params.mode = NudgedStaticMode::Relative;
                is >> params.relative_nudge;
            }
            else if (token == "interpolate")
            {
                params.mode = NudgedStaticMode::Interpolate;
                is >> params.interpolate_nudge;
            }
            else if (token == "input_file")
                is >> params.input_filename;
            else if (token == "output_file")
                is >> params.output_filename;
            else
            {
                std::cout << "ERROR: Unknown option " << token << ". Exiting...\n";
                return;
            }
        }

        std::cout << "Performing transform nudged_static with parameters:\n";
        std::cout << "input_file          : " << params.input_filename << '\n';
        std::cout << "output_file         : " << params.output_filename << '\n';
        std::cout << "\n";
        if (params.mode == NudgedStaticMode::Absolute)
        {
            std::cout << "mode                : absolute\n";
            std::cout << "absolute_nudge      : " << params.absolute_nudge << '\n';
        }
        else if (params.mode == NudgedStaticMode::Relative)
        {
            std::cout << "mode                : relative\n";
            std::cout << "relative_nudge      : " << params.relative_nudge << '\n';
        }
        else if (params.mode == NudgedStaticMode::Interpolate)
        {
            std::cout << "mode                : interpolate\n";
            std::cout << "interpolate_nudge   : " << params.interpolate_nudge << '\n';
        }
        std::cout << '\n';

        params.enforce_constraints();
        do_nudged_static(params);
    }

    void do_rescore_epd(RescoreParams& params)
    {
        std::ifstream fens_file(params.input_filename);

        auto next_fen = [&fens_file, mutex = std::mutex{}]() mutable -> std::optional<std::string>{
            std::string fen;

            std::unique_lock lock(mutex);

            if (std::getline(fens_file, fen) && fen.size() >= 10)
            {
                return fen;
            }
            else
            {
                return std::nullopt;
            }
        };

        PSVector buffer;
        uint64_t batch_size = 10'000;

        buffer.reserve(batch_size);

        auto out = Tools::create_new_sfen_output(params.output_filename);

        std::mutex mutex;
        uint64_t num_processed = 0;

        // About Search::Limits
        // Be careful because this member variable is global and affects other threads.
        auto& limits = Search::Limits;

        // Make the search equivalent to the "go infinite" command. (Because it is troublesome if time management is done)
        limits.infinite = true;

        // Since PV is an obstacle when displayed, erase it.
        limits.silent = true;

        // If you use this, it will be compared with the accumulated nodes of each thread. Therefore, do not use it.
        limits.nodes = 0;

        // depth is also processed by the one passed as an argument of Tools::search().
        limits.depth = 0;

        Threads.execute_with_workers([&](auto& th){
            Position& pos = th.rootPos;
            StateInfo si;

            for(;;)
            {
                auto fen = next_fen();
                if (!fen.has_value())
                    return;

                pos.set(variants.find(Options["UCI_Variant"])->second, *fen, false, &si, &th);
                pos.state()->rule50 = 0;


                for (int cnt = 0; cnt < params.research_count; ++cnt)
                    Search::search(pos, params.depth, 1);

                auto [search_value, search_pv] = Search::search(pos, params.depth, 1);

                if (search_pv.empty())
                    continue;

                PackedSfenValue ps;
                pos.sfen_pack(ps.sfen);
                ps.score = search_value;
                ps.move = search_pv[0];
                ps.gamePly = 1;
                ps.game_result = 0;
                ps.padding = 0;

                std::unique_lock lock(mutex);
                buffer.emplace_back(ps);
                if (buffer.size() >= batch_size)
                {
                    num_processed += buffer.size();

                    out->write(buffer);
                    buffer.clear();

                    std::cout << "Processed " << num_processed << " positions.\n";
                }
            }
        });
        Threads.wait_for_workers_finished();

        if (!buffer.empty())
        {
            num_processed += buffer.size();

            out->write(buffer);
            buffer.clear();

            std::cout << "Processed " << num_processed << " positions.\n";
        }

        std::cout << "Finished.\n";
    }

    void do_rescore_data(RescoreParams& params)
    {
        // TODO: Use SfenReader once it works correctly in sequential mode. See issue #271
        auto in = Tools::open_sfen_input_file(params.input_filename);
        auto readsome = [&in, mutex = std::mutex{}](int n) mutable -> PSVector {

            PSVector psv;
            psv.reserve(n);

            std::unique_lock lock(mutex);

            for (int i = 0; i < n; ++i)
            {
                auto ps_opt = in->next();
                if (ps_opt.has_value())
                {
                    psv.emplace_back(*ps_opt);
                }
                else
                {
                    break;
                }
            }

            return psv;
        };

        auto sfen_format = SfenOutputType::Bin;

        auto out = SfenWriter(
            params.output_filename,
            Threads.size(),
            std::numeric_limits<std::uint64_t>::max(),
            sfen_format);

        // About Search::Limits
        // Be careful because this member variable is global and affects other threads.
        auto& limits = Search::Limits;

        // Make the search equivalent to the "go infinite" command. (Because it is troublesome if time management is done)
        limits.infinite = true;

        // Since PV is an obstacle when displayed, erase it.
        limits.silent = true;

        // If you use this, it will be compared with the accumulated nodes of each thread. Therefore, do not use it.
        limits.nodes = 0;

        // depth is also processed by the one passed as an argument of Tools::search().
        limits.depth = 0;

        std::atomic<std::uint64_t> num_processed = 0;

        Threads.execute_with_workers([&](auto& th){
            Position& pos = th.rootPos;
            StateInfo si;

            for (;;)
            {
                PSVector psv = readsome(5000);
                if (psv.empty())
                    break;

                for(auto& ps : psv)
                {
                    pos.set_from_packed_sfen(ps.sfen, &si, &th);

                    for (int cnt = 0; cnt < params.research_count; ++cnt)
                        Search::search(pos, params.depth, 1);

                    auto [search_value, search_pv] = Search::search(pos, params.depth, 1);

                    if (search_pv.empty())
                        continue;

                    pos.sfen_pack(ps.sfen);
                    ps.score = search_value;
                    if (!params.keep_moves)
                        ps.move = search_pv[0];
                    ps.padding = 0;

                    out.write(th.id(), ps);

                    auto p = num_processed.fetch_add(1) + 1;
                    if (p % 10000 == 0)
                    {
                        std::cout << "Processed " << p << " positions.\n";
                    }
                }
            }
        });
        Threads.wait_for_workers_finished();

        std::cout << "Finished.\n";
    }

    void do_rescore(RescoreParams& params)
    {
        if (ends_with(params.input_filename, ".epd"))
        {
            do_rescore_epd(params);
        }
        else if (ends_with(params.input_filename, ".bin"))
        {
            do_rescore_data(params);
        }
        else
        {
            std::cerr << "Invalid input file type.\n";
        }
    }

    void rescore(std::istringstream& is)
    {
        RescoreParams params{};

        while(true)
        {
            std::string token;
            is >> token;

            if (token == "")
                break;

            if (token == "depth")
                is >> params.depth;
            else if (token == "input_file")
                is >> params.input_filename;
            else if (token == "output_file")
                is >> params.output_filename;
            else if (token == "keep_moves")
                is >> params.keep_moves;
            else if (token == "research_count")
                is >> params.research_count;
            else
            {
                std::cout << "ERROR: Unknown option " << token << ". Exiting...\n";
                return;
            }
        }

        params.enforce_constraints();

        std::cout << "Performing transform rescore with parameters:\n";
        std::cout << "depth               : " << params.depth << '\n';
        std::cout << "input_file          : " << params.input_filename << '\n';
        std::cout << "output_file         : " << params.output_filename << '\n';
        std::cout << "keep_moves          : " << params.keep_moves << '\n';
        std::cout << "research_count      : " << params.research_count << '\n';
        std::cout << '\n';

        do_rescore(params);
    }

    void transform(std::istringstream& is)
    {
        const std::map<std::string, CommandFunc> subcommands = {
            { "nudged_static", &nudged_static },
            { "rescore", &rescore },
            { "mine", &mine }
        };

        Eval::NNUE::init();

        std::string subcommand;
        is >> subcommand;

        auto func = subcommands.find(subcommand);
        if (func == subcommands.end())
        {
            std::cout << "Invalid subcommand " << subcommand << ". Exiting...\n";
            return;
        }

        func->second(is);
    }

}
