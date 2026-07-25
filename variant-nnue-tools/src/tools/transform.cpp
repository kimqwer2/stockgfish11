#include "transform.h"

#include "sfen_stream.h"
#include "packed_sfen.h"
#include "sfen_writer.h"

#include "thread.h"
#include "position.h"
#include "evaluate.h"
#include "misc.h"
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
#include <filesystem>
#include <utility>

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


    struct RescoreParams
    {
        std::string input_filename = "in.epd";
        std::string output_filename = "out.bin";
        int depth = 3;
        std::uint64_t nodes = 0;
        int research_count = 0;
        bool keep_moves = true;

        void enforce_constraints()
        {
            depth = std::max(1, depth);
            nodes = std::max<std::uint64_t>(0, nodes);
            research_count = std::max(0, research_count);
        }
    };

    void do_rescore(RescoreParams& params);

    struct MineParams
    {
        std::string mode = "search-gap";
        std::string input_filename = "in.bin";
        std::string output_filename = "hard.bin";
        std::string net;
        std::string old_net;
        std::string new_net;
        std::string variant;
        std::uint64_t keep_count = 1000000;
        std::uint64_t batch_size = 100000;
        int min_gap = 0;
        int depth = 3;
        std::uint64_t nodes = 0;
        int research_count = 0;
        bool keep_moves = true;
    };


    struct ScopedTempFile
    {
        explicit ScopedTempFile(std::string filename_) : filename(std::move(filename_)) {}
        ~ScopedTempFile()
        {
            if (!filename.empty())
                std::remove(filename.c_str());
        }
        void dismiss() { filename.clear(); }
        std::string filename;
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

    static std::string path_basename(const std::string& path)
    {
        const auto pos = path.find_last_of("\\/");
        return pos == std::string::npos ? path : path.substr(pos + 1);
    }

    static std::string infer_variant_from_eval_file(const std::string& filename)
    {
        const std::string basename = path_basename(filename);
        for (const auto& [name, variant] : variants)
        {
            if (basename.rfind(name, 0) == 0)
                return name;
            if (!variant->nnueAlias.empty() && basename.rfind(variant->nnueAlias, 0) == 0)
                return name;
        }
        return std::string(Options["UCI_Variant"]);
    }

    static std::string absolute_path_for_diagnostics(const std::string& filename)
    {
        try
        {
            return std::filesystem::absolute(std::filesystem::path(filename)).string();
        }
        catch (const std::exception& e)
        {
            return std::string("<absolute path unavailable: ") + e.what() + ">";
        }
    }

    static std::string exists_for_diagnostics(const std::string& filename)
    {
        try
        {
            return std::filesystem::exists(std::filesystem::path(filename)) ? "yes" : "no";
        }
        catch (const std::exception& e)
        {
            return std::string("unknown: ") + e.what();
        }
    }

    static void print_nnue_load_diagnostics(const std::string& requested_filename,
                                            const std::string& selected_variant,
                                            const std::string& loaded_before)
    {
        std::cerr << "ERROR: Could not load NNUE via Eval::NNUE::init()\n"
                  << "  requested EvalFile       : " << requested_filename << "\n"
                  << "  requested absolute path  : " << absolute_path_for_diagnostics(requested_filename) << "\n"
                  << "  requested path exists    : " << exists_for_diagnostics(requested_filename) << "\n"
                  << "  executable-dir candidate : " << CommandLine::binaryDirectory + requested_filename << "\n"
                  << "  executable candidate exists: " << exists_for_diagnostics(CommandLine::binaryDirectory + requested_filename) << "\n"
                  << "  working directory        : " << CommandLine::workingDirectory << "\n"
                  << "  executable directory     : " << CommandLine::binaryDirectory << "\n"
                  << "  UCI_Variant              : " << std::string(Options["UCI_Variant"]) << "\n"
                  << "  inferred/selected variant: " << selected_variant << "\n"
                  << "  EvalFile option          : " << std::string(Options["EvalFile"]) << "\n"
                  << "  previously loaded NNUE   : " << loaded_before << "\n"
                  << "  currently loaded NNUE    : " << Eval::NNUE::eval_file_loaded << "\n"
                  << "  loader                   : engine EvalFile path (Eval::NNUE::init)\n"
                  << "  search locations         : current working directory, executable directory, and DEFAULT_NNUE_DIRECTORY if compiled\n"
                  << "  common causes            : incompatible network architecture, wrong UCI_Variant for this net, unsupported/corrupt NNUE format, or path quoting/working-directory mismatch\n";
    }

    static bool load_nnue_file(const std::string& filename, const std::string& requested_variant)
    {
        const std::string variant = requested_variant.empty() ? infer_variant_from_eval_file(filename) : requested_variant;
        if (!variants.count(variant))
        {
            std::cerr << "ERROR: Unknown variant for NNUE load: " << variant << "\n"
                      << "  requested EvalFile      : " << filename << "\n"
                      << "  current UCI_Variant     : " << std::string(Options["UCI_Variant"]) << "\n";
            return false;
        }

        const std::string loaded_before = Eval::NNUE::eval_file_loaded;

        Options["Use NNUE"] = std::string("true");
        if (std::string(Options["UCI_Variant"]) != variant)
            Options["UCI_Variant"] = variant;

        // Use the same engine loading path as the EvalFile UCI option. This keeps
        // working-directory, executable-directory, embedded/default-directory,
        // format validation, and network lifetime behavior identical to engine eval.
        Options["EvalFile"] = filename;
        Eval::NNUE::init();

        if (Eval::NNUE::useNNUE == Eval::NNUE::UseNNUEMode::False || Eval::NNUE::eval_file_loaded != filename)
        {
            print_nnue_load_diagnostics(filename, variant, loaded_before);
            return false;
        }

        return true;
    }

    static void add_mined_entry(std::priority_queue<MinedEntry, std::vector<MinedEntry>, std::greater<MinedEntry>>& heap,
                                std::uint64_t keep_count,
                                int min_gap,
                                int gap,
                                std::uint64_t ordinal,
                                const PackedSfenValue& psv)
    {
        if (keep_count == 0 || gap < min_gap)
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
        const auto write_count = heap.size();

        std::remove(Tools::filename_with_extension(output_filename, Tools::BinSfenOutputStream::extension).c_str());
        auto out = Tools::create_new_sfen_output(output_filename);
        if (out == nullptr)
        {
            std::cerr << "Invalid output file type.\n";
            return;
        }

        PSVector buffer;
        buffer.reserve(batch_size);
        while (!heap.empty())
        {
            buffer.emplace_back(heap.top().psv);
            heap.pop();
            if (buffer.size() >= batch_size)
            {
                out->write(buffer);
                buffer.clear();
            }
        }
        if (!buffer.empty())
            out->write(buffer);

        std::cout << "Wrote " << write_count << " mined positions to " << output_filename << "\n";
    }

    static void do_mine_search_gap(MineParams& params)
    {
        if (!load_nnue_file(params.net, params.variant))
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
            add_mined_entry(heap, params.keep_count, params.min_gap, gap, processed, psv);
            if (++processed % 1000000 == 0)
                std::cout << "Processed " << processed << " positions. Current cutoff gap " << (heap.empty() ? 0 : heap.top().gap) << "\n";
        }
        write_mined_entries(heap, params.output_filename, params.batch_size);
    }

    static void do_mine_eval_disagree(MineParams& params)
    {
        const std::string temp_filename = params.output_filename + ".old_eval.tmp";
        ScopedTempFile temp_file(temp_filename);
        {
            if (!load_nnue_file(params.old_net, params.variant))
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

        if (!load_nnue_file(params.new_net, params.variant))
            return;
        Thread* th = Threads.main();
        Position& pos = th->rootPos;
        StateInfo si;
        auto in = Tools::open_sfen_input_file(params.input_filename);
        std::ifstream temp(temp_filename, std::ios::binary);
        if (in == nullptr || !temp)
        {
            std::cerr << "Invalid input file type or temporary file.\n";
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
            add_mined_entry(heap, params.keep_count, params.min_gap, gap, processed, psv);
            if (++processed % 1000000 == 0)
                std::cout << "New net pass processed " << processed << " positions. Current cutoff gap " << (heap.empty() ? 0 : heap.top().gap) << "\n";
        }
        write_mined_entries(heap, params.output_filename, params.batch_size);
    }

    static void do_mine_search_gap_deep(MineParams& params)
    {
        const std::string temp_filename = params.output_filename + ".mined.tmp.bin";
        MineParams mine_params = params;
        mine_params.output_filename = temp_filename;

        const std::string temp_bin = Tools::filename_with_extension(temp_filename, Tools::BinSfenOutputStream::extension);
        ScopedTempFile temp_file(temp_bin);
        std::remove(temp_bin.c_str());
        do_mine_search_gap(mine_params);
        {
            std::ifstream mined_temp(temp_bin, std::ios::binary);
            if (!mined_temp)
            {
                std::cerr << "ERROR: search-gap-deep mining pass did not produce " << temp_bin << "\n";
                return;
            }
        }

        RescoreParams rescore_params;
        rescore_params.input_filename = temp_filename;
        rescore_params.output_filename = params.output_filename;
        rescore_params.depth = params.depth;
        rescore_params.nodes = params.nodes;
        rescore_params.research_count = params.research_count;
        rescore_params.keep_moves = params.keep_moves;
        rescore_params.enforce_constraints();

        std::remove(Tools::filename_with_extension(params.output_filename, Tools::BinSfenOutputStream::extension).c_str());
        do_rescore(rescore_params);
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
            else if (token == "variant" || token == "--variant") is >> params.variant;
            else if (token == "keep_count" || token == "--keep-count") is >> params.keep_count;
            else if (token == "batch_size" || token == "--batch-size") is >> params.batch_size;
            else if (token == "min_gap" || token == "--min-gap" || token == "threshold" || token == "--threshold") is >> params.min_gap;
            else if (token == "depth" || token == "--depth") is >> params.depth;
            else if (token == "nodes" || token == "--nodes") is >> params.nodes;
            else if (token == "research_count" || token == "--research-count") is >> params.research_count;
            else if (token == "keep_moves" || token == "--keep-moves") is >> params.keep_moves;
            else { std::cerr << "ERROR: Unknown option " << token << "\n"; return; }
        }
        params.batch_size = std::max<std::uint64_t>(1, params.batch_size);
        params.min_gap = std::max(0, params.min_gap);
        params.depth = std::max(1, params.depth);
        params.research_count = std::max(0, params.research_count);
        std::cout << "Hard position mining mode=" << params.mode << " input=" << params.input_filename << " output=" << params.output_filename << " keep_count=" << params.keep_count << " min_gap=" << params.min_gap
                  << " variant=" << (params.variant.empty() ? infer_variant_from_eval_file(params.mode == "eval-disagree" ? params.new_net : params.net) : params.variant) << "\n";
        if (params.mode == "search-gap")
        {
            if (params.net.empty()) { std::cerr << "ERROR: search-gap requires --net.\n"; return; }
            do_mine_search_gap(params);
        }
        else if (params.mode == "search-gap-deep")
        {
            if (params.net.empty()) { std::cerr << "ERROR: search-gap-deep requires --net.\n"; return; }
            do_mine_search_gap_deep(params);
        }
        else if (params.mode == "eval-disagree")
        {
            if (params.old_net.empty() || params.new_net.empty()) { std::cerr << "ERROR: eval-disagree requires --old-net and --new-net.\n"; return; }
            do_mine_eval_disagree(params);
        }
        else std::cerr << "ERROR: Unsupported mining mode " << params.mode << "\n";
    }

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
            else if (token == "input_file" || token == "--input")
                is >> params.input_filename;
            else if (token == "output_file" || token == "--output")
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
                    Search::search(pos, params.depth, 1, params.nodes);

                auto [search_value, search_pv] = Search::search(pos, params.depth, 1, params.nodes);

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
                        Search::search(pos, params.depth, 1, params.nodes);

                    auto [search_value, search_pv] = Search::search(pos, params.depth, 1, params.nodes);

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

            if (token == "depth" || token == "--depth")
                is >> params.depth;
            else if (token == "nodes" || token == "--nodes")
                is >> params.nodes;
            else if (token == "input_file" || token == "--input")
                is >> params.input_filename;
            else if (token == "output_file" || token == "--output")
                is >> params.output_filename;
            else if (token == "keep_moves" || token == "--keep-moves")
                is >> params.keep_moves;
            else if (token == "research_count" || token == "--research-count")
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
        std::cout << "nodes               : " << params.nodes << '\n';
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
