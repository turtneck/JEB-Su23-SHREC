#include "puzzle_solver.hpp"

#include <functional>
#include <unordered_set>

#include "puzzle.hpp"
#include "state.hpp"
#include "frontier_queue.hpp"

// specialize the std::hash function for Puzzles
namespace std {
template <> struct hash<Puzzle> {
  typedef Puzzle argument_type;
  typedef uint64_t result_type;

  result_type operator()(const argument_type &x) const { return x.hash(); }
};
}

PuzzleSolver::PuzzleSolver(const Puzzle &initialPuzzle,
                           const Puzzle &goalPuzzle)
    : initial(initialPuzzle), goal(goalPuzzle) {}

std::pair<bool, std::size_t> PuzzleSolver::search() {

  std::size_t solution_cost;

  frontier_queue<Puzzle> frontier;
  std::unordered_set<Puzzle> explored;

  frontier.push(initial, 0, initial.heuristic(goal));

  bool found = false;
  while (true) {
    if (frontier.empty()) {
      found = false;
      break;
    }

    auto node = frontier.pop();
    Puzzle value = node.getValue();
    explored.insert(value);

    std::size_t cost = node.getPathCost();

    if (value.heuristic(goal) == 0) {
      found = true;
      solution_cost = node.getPathCost();
      break;
    }

    for (int a = Puzzle::LEFT; a <= Puzzle::DOWN; ++a) {
      bool ok;
      Puzzle result;
      std::tie(ok, result) = value.apply(static_cast<Puzzle::Action>(a));

      if (ok) {
        bool inexplored = explored.count(result) != 0;
        bool infrontier = frontier.contains(result);
        if (!(inexplored || infrontier)) {
          frontier.push(result, cost + 1, result.heuristic(goal));
        } else if (infrontier) {
          frontier.replaceif(result, cost + 1);
        }
      }
    }
  }

  return std::make_pair(found, solution_cost);
}
