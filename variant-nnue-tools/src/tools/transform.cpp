#include "transform.h"

#include "sfen_stream.h"
#include "packed_sfen.h"
#include "sfen_writer.h"

#include "thread.h"
#include "position.h"
#include "movegen.h"
#include "evaluate.h"
#include "misc.h"
#include "search.h"
#include "uci.h"

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
#include <chrono>
#include <csignal>
#include <iomanip>
#include <sstream>

namespace Stockfish::Tools
{
    // ★ 탐색 도중 발생하는 std::cout (info depth ...) 콘솔 출력을 완전 차단하는 RAII 클래스 ★
    struct ScopedSilenceCout
    {
        std::streambuf* old_buf;
        std::ofstream null_stream;

        ScopedSilenceCout()
        {
#ifdef _WIN32
            null_stream.open("NUL");
#else
            null_stream.open("/dev/null");
#endif
            old_buf = std::cout.rdbuf(null_stream.rdbuf());
        }

        ~ScopedSilenceCout()
        {
            std::cout.rdbuf(old_buf);
        }
    };

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
        std::string net;
        std::string variant;
        int depth = 3;
        int threads = 1;
        std::uint64_t nodes = 0;
        int research_count = 0;
        bool keep_moves = true;
        std::uint64_t log_interval = 1000;
        std::uint64_t sample_count = 10;
        std::uint64_t flush_interval = 1000;

        void enforce_constraints()
        {
            depth = std::max(1, depth);
            threads = std::max(1, threads);
            nodes = std::max<std::uint64_t>(0, nodes);
            research_count = std::max(0, research_count);
            log_interval = std::max<std::uint64_t>(1, log_interval);
            flush_interval = std::max<std::uint64_t>(1, flush_interval);
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

    static std::string absolute_path_for_diagnostics(const std::string& path)
    {
        std::error_code ec;
        auto p = std::filesystem::absolute(path, ec);

        if (ec)
            return "<absolute path unavailable>";

        return p.string();
    }

    static std::string exists_for_diagnostics(const std::string& path)
    {
        std::error_code ec;
        bool exists = std::filesystem::exists(path, ec);

        if (ec)
            return "unknown";

        return exists ? "yes" : "no";
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
                out.reset();
                out = Tools::create_new_sfen_output(output_filename);
                if (out == nullptr)
                {
                    std::cerr << "ERROR: Failed to reopen output after flush.\n";
                    return;
                }
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
        rescore_params.net = params.net;
        rescore_params.variant = params.variant;
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


    static std::uint64_t estimate_packed_position_count(const std::string& input_filename)
    {
        const std::string bin = Tools::filename_with_extension(input_filename, Tools::BinSfenOutputStream::extension);
        std::error_code ec;
        if (!std::filesystem::exists(bin, ec)) return 0;
        const auto bytes = std::filesystem::file_size(bin, ec);
        return ec ? 0 : bytes / sizeof(PackedSfenValue);
    }

    static Search::ValueAndPV threaded_search(Position& pos, const RescoreParams& params, int& completed_depth, std::uint64_t& nodes)
    {
        StateListPtr states(new std::deque<StateInfo>(1));
        states->back() = *pos.state();
        Search::LimitsType limits;
        limits.depth = params.depth;
        limits.nodes = static_cast<int64_t>(params.nodes);
        limits.silent = true;
        Threads.start_thinking(pos, states, limits, false);
        Threads.main()->wait_for_search_finished();
        Thread* best = Threads.main()->rootMoves.empty() ? Threads.main() : Threads.get_best_thread();
        completed_depth = best->completedDepth;
        nodes = Threads.nodes_searched();
        if (best->rootMoves.empty()) return {};
        return {best->rootMoves[0].score, best->rootMoves[0].pv};
    }


    static std::string g_rescore_checkpoint_filename;
    static std::uint64_t g_rescore_current_index = 0;
    static PackedSfenValue g_rescore_current_ps{};

    static std::string packed_sfen_to_hex(const PackedSfen& sfen)
    {
        std::ostringstream os;
        os << std::hex << std::setfill('0');
        for (std::uint8_t byte : sfen.data)
            os << std::setw(2) << unsigned(byte);
        return os.str();
    }

    static void write_rescore_checkpoint(std::uint64_t index, const PackedSfenValue& ps, const std::string& note)
    {
        if (g_rescore_checkpoint_filename.empty())
            return;
        std::ofstream checkpoint(g_rescore_checkpoint_filename, std::ios::trunc);
        if (!checkpoint)
            return;
        checkpoint << "note: " << note << "\n"
                   << "position_index: " << index << "\n"
                   << "score: " << ps.score << "\n"
                   << "move: " << ps.move << "\n"
                   << "gamePly: " << ps.gamePly << "\n"
                   << "game_result: " << int(ps.game_result) << "\n"
                   << "packed_sfen_hex: " << packed_sfen_to_hex(ps.sfen) << "\n";
    }

    static void rescore_signal_handler(int signal)
    {
        write_rescore_checkpoint(g_rescore_current_index, g_rescore_current_ps, std::string("fatal signal ") + std::to_string(signal));
        std::_Exit(128 + signal);
    }

    static void install_rescore_signal_handlers(const std::string& checkpoint_filename)
    {
        g_rescore_checkpoint_filename = checkpoint_filename;
        std::signal(SIGABRT, rescore_signal_handler);
        std::signal(SIGSEGV, rescore_signal_handler);
#ifdef SIGBUS
        std::signal(SIGBUS, rescore_signal_handler);
#endif
    }

    void do_rescore_epd(RescoreParams& params)
    {
        std::ifstream fens_file(params.input_filename);
        if (!fens_file.is_open())
        {
            std::cerr << "ERROR: Could not open input file " << params.input_filename << "\n";
            return;
        }

        auto out = Tools::create_new_sfen_output(params.output_filename);
        if (out == nullptr)
        {
            std::cerr << "ERROR: Invalid output file type.\n";
            return;
        }

        Thread* th = Threads.main();
        Position& pos = th->rootPos;
        StateInfo si;

        PSVector buffer;
        const uint64_t batch_size = params.flush_interval;
        buffer.reserve(batch_size);

        uint64_t num_processed = 0;
        std::string fen;

        while (std::getline(fens_file, fen))
        {
            if (fen.size() < 10)
                continue;

            pos.set(variants.find(Options["UCI_Variant"])->second, fen, false, &si, th);
            pos.state()->rule50 = 0;

            Value search_value;
            std::vector<Move> search_pv;

            // ★ 탐색 중에만 콘솔 출력을 차단하고 지정한 threads 사용 ★
            {
                ScopedSilenceCout silence;
                for (int cnt = 0; cnt < params.research_count; ++cnt)
                    Search::search(pos, params.depth, params.threads, params.nodes);

                std::tie(search_value, search_pv) = Search::search(pos, params.depth, params.threads, params.nodes);
            }

            if (search_pv.empty())
                continue;

            PackedSfenValue ps{};
            pos.sfen_pack(ps.sfen);
            ps.score = static_cast<std::int16_t>(search_value);
            ps.move = search_pv[0];
            ps.gamePly = 1;
            ps.game_result = 0;
            ps.padding = 0;

            buffer.emplace_back(ps);
            num_processed++;

            if (num_processed % 10000 == 0)
            {
                std::cout << "Processed " << num_processed << " positions.\n";
            }

            if (buffer.size() >= batch_size)
            {
                out->write(buffer);
                buffer.clear();
            }
        }

        if (!buffer.empty())
        {
            out->write(buffer);
            buffer.clear();
        }

        std::cout << "Finished rescoring. Total processed: " << num_processed << " positions.\n";
    }

    void do_rescore_data(RescoreParams& params)
    {
        auto in = Tools::open_sfen_input_file(params.input_filename);
        if (in == nullptr)
        {
            std::cerr << "ERROR: Invalid input file type or failed to open " << params.input_filename << "\n";
            return;
        }
        std::remove(Tools::filename_with_extension(params.output_filename, Tools::BinSfenOutputStream::extension).c_str());
        auto out = Tools::create_new_sfen_output(params.output_filename);
        if (out == nullptr)
        {
            std::cerr << "ERROR: Invalid output file type.\n";
            return;
        }

        Thread* th = Threads.main();
        Position& pos = th->rootPos;
        StateInfo si;
        PSVector buffer;
        const uint64_t batch_size = params.flush_interval;
        buffer.reserve(batch_size);

        uint64_t num_processed = 0, pv_changed = 0, total_nodes = 0, total_depth = 0, total_ms = 0;
        install_rescore_signal_handlers(params.output_filename + ".checkpoint.txt");
        std::cout << "Rescore checkpoint file: " << g_rescore_checkpoint_filename << "\n"
                  << "Rescore input position estimate: " << estimate_packed_position_count(params.input_filename) << "\n"
                  << "Rescore depth: " << params.depth << " nodes: " << params.nodes
                  << " Threads: " << Threads.size() << " loaded NNUE: " << Eval::NNUE::eval_file_loaded << "\n";

        for (;;)
        {
            auto v = in->next();
            if (!v.has_value())
                break;
            auto ps = v.value();
            const auto current_index = num_processed + 1;
            g_rescore_current_index = current_index;
            g_rescore_current_ps = ps;
            write_rescore_checkpoint(current_index, ps, "before search");
            const auto old_score = ps.score;
            const auto old_move = Move(ps.move);
            pos.set_from_packed_sfen(ps.sfen, &si, th);

            Value search_value;
            std::vector<Move> search_pv;
            int completed_depth = 0;
            std::uint64_t search_nodes = 0;
            const auto start_time = std::chrono::steady_clock::now();
            {
                ScopedSilenceCout silence;
                for (int cnt = 0; cnt < params.research_count; ++cnt)
                    threaded_search(pos, params, completed_depth, search_nodes);
                std::tie(search_value, search_pv) = threaded_search(pos, params, completed_depth, search_nodes);
            }
            if (search_pv.empty())
                continue;
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
            total_ms += elapsed_ms; total_nodes += search_nodes; total_depth += completed_depth;

            ps.score = static_cast<std::int16_t>(search_value);
            if (old_move != search_pv[0])
                ++pv_changed;
            if (!params.keep_moves)
                ps.move = search_pv[0];
            ps.padding = 0;
            buffer.emplace_back(ps);
            ++num_processed;

            if (num_processed <= params.sample_count)
                std::cout << "Sample " << num_processed << " index: " << current_index << " old score: " << old_score << " new score: " << ps.score
                          << " old move: " << UCI::move(pos, old_move) << " new move: " << UCI::move(pos, search_pv[0])
                          << " completed depth: " << completed_depth << " nodes: " << search_nodes << " time ms: " << elapsed_ms << "\n";
            if (num_processed % params.log_interval == 0)
                std::cout << "Processed " << num_processed << " last index: " << current_index
                          << " avg nodes: " << total_nodes / num_processed
                          << " avg depth: " << double(total_depth) / double(num_processed)
                          << " avg search time ms: " << total_ms / num_processed
                          << " PV changed count: " << pv_changed << "\n";

            if (buffer.size() >= batch_size)
            {
                out->write(buffer);
                buffer.clear();
                out.reset();
                out = Tools::create_new_sfen_output(params.output_filename);
                if (out == nullptr)
                {
                    std::cerr << "ERROR: Failed to reopen output after flush.\n";
                    return;
                }
            }
        }
        if (!buffer.empty())
            out->write(buffer);
        std::cout << "Finished rescoring. Total processed: " << num_processed << " positions. PV changed count: " << pv_changed << "\n";
    }

    void do_rescore(RescoreParams& params)
    {
        // 1. NNUE 모델 및 Variant 로드
        if (!params.net.empty())
        {
            if (!load_nnue_file(params.net, params.variant))
                return;
        }
        else if (!params.variant.empty())
        {
            if (variants.count(params.variant))
                Options["UCI_Variant"] = params.variant;
        }

        // 2. Stockfish 내부 탐색 스레드 설정
        Options["Threads"] = std::to_string(params.threads);
        Threads.set(params.threads);

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
            else if (token == "threads" || token == "--threads")
                is >> params.threads;
            else if (token == "nodes" || token == "--nodes")
                is >> params.nodes;
            else if (token == "input_file" || token == "--input")
                is >> params.input_filename;
            else if (token == "output_file" || token == "--output")
                is >> params.output_filename;
            else if (token == "net" || token == "--net")
                is >> params.net;
            else if (token == "variant" || token == "--variant")
                is >> params.variant;
            else if (token == "keep_moves" || token == "--keep-moves")
                is >> params.keep_moves;
            else if (token == "research_count" || token == "--research-count")
                is >> params.research_count;
            else if (token == "log_interval" || token == "--log-interval")
                is >> params.log_interval;
            else if (token == "sample_count" || token == "--sample-count")
                is >> params.sample_count;
            else if (token == "flush_interval" || token == "--flush-interval")
                is >> params.flush_interval;
            else
            {
                std::cout << "ERROR: Unknown option " << token << ". Exiting...\n";
                return;
            }
        }

        params.enforce_constraints();

        std::cout << "Performing transform rescore with parameters:\n";
        std::cout << "depth               : " << params.depth << '\n';
        std::cout << "threads             : " << params.threads << '\n';
        std::cout << "nodes               : " << params.nodes << '\n';
        std::cout << "input_file          : " << params.input_filename << '\n';
        std::cout << "output_file         : " << params.output_filename << '\n';
        std::cout << "net                 : " << params.net << '\n';
        std::cout << "variant             : " << params.variant << '\n';
        std::cout << "keep_moves          : " << params.keep_moves << '\n';
        std::cout << "research_count      : " << params.research_count << '\n';
        std::cout << "log_interval        : " << params.log_interval << '\n';
        std::cout << "sample_count        : " << params.sample_count << '\n';
        std::cout << "flush_interval       : " << params.flush_interval << '\n';
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