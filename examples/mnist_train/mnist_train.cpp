#include <iostream>
#include <string>
#include <gflags/gflags.h>
#include <caffe/caffe.hpp>
#include <unordered_map>
#include <deque>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

//Common flags
DEFINE_int32(train, 1, "Whether we want to train the net or test it");
DEFINE_int32(board_size, 6, "Size of the board");
DEFINE_int32(aligned_to_win, 4, "Number of aligned token to win the game");
DEFINE_int32(num_playouts, 400, "Number of playouts computed by the MCTS algorithm at each move");
DEFINE_double(c_puct, 3.0, "Exploration constant used by MCTS algorithm");
DEFINE_int32(num_test, 25, "Number of games used for testing (or at the end of each epoch during training)");

//Training flags
DEFINE_string(solver_file, "solver.prototxt", "Solver file for training");
DEFINE_int32(batch_size, 2, "Batch size used for training");
DEFINE_double(alpha, 0.3, "Dirichlet noise used during self play");
DEFINE_int32(max_size_memory, 10000, "Maximum size of the memory buffer");
DEFINE_int32(num_game, 10000, "Number of self-played games");
DEFINE_string(log_file, "log.csv", "Name of the file to log the losses during training");
DEFINE_double(epsilon, 0.25, "Dirichlet noise weight");
DEFINE_string(snapshot, "", "Solverstate file to restart training");

//Testing flags
DEFINE_string(model_file, "net_tic_tac_6_4.prototxt", "Prototxt file describing the network architecture");
DEFINE_string(model_file2, "", "Prototxt file describing the network architecture to compare against the first");
DEFINE_string(trained_weights, "", "Trained net to load into the the network");
DEFINE_string(trained_weights2, "", "Trained net to load into the the second network");
DEFINE_int32(opponent_playouts, 1000, "Number of playouts for the opponent (only used with MCTS training mode)");
DEFINE_string(test_mode, "Human", "Which kind of test you want to perform (Human, Self, Compare, MCTS, Random)");
DEFINE_int32(display, 0, "Whether the intermediate states should be displayed durin testing");

int main(int argc, char** argv) 
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);

	//In test mode we don't need all the caffe stuff
	if (FLAGS_train)
	{
		google::LogToStderr();
	}
	else
	{
		for (int i = 0; i < google::NUM_SEVERITIES; ++i)
		{
			google::SetLogDestination(i, "");
		}
	}

	return 0;
}
