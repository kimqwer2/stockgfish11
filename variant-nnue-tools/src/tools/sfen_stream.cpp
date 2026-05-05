#include "sfen_stream.h"
#include "position.h"
#include "thread.h"

namespace Stockfish::Tools {

    EpdSfenOutputStream::EpdSfenOutputStream(std::string filename)
        : m_stream(filename_with_extension(filename, extension), openmode)
    {
    }

    void EpdSfenOutputStream::write(const PSVector& sfens)
    {
        // We need a Position and Thread to unpack the sfens
        // Get the main thread for unpacking positions
        auto th = Threads.main();
        Position pos;
        StateInfo si;

        for (const auto& psv : sfens)
        {
            set_from_packed_sfen(pos, psv.sfen, &si, th);

            std::string fen_line = pos.fen();
            fen_line.push_back('\n');
            m_stream << fen_line;
        }
    }

}
