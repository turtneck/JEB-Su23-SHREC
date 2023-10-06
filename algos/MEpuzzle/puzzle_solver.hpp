#ifndef PUZZLE_SOLVER_HPP
#define PUZZLE_SOLVER_HPP

#include <utility>

#include "puzzle.hpp"

class PuzzleSolver{
public:

  PuzzleSolver(const Puzzle & initialPuzzle, const Puzzle & goalPuzzle);

  std::pair<bool, std::size_t> search();
  
private:

  Puzzle initial, goal;
};

#endif
