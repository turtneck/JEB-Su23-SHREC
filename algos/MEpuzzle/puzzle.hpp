#ifndef PUZZLE_HPP
#define PUZZLE_HPP

#include <utility>
#include <string>
#include <sstream>

// A puzzle is a 3x3 array of cells containing a label
// indexed by row, and column
class Puzzle {
public:

  // tiles are labeled A-H and BLANK
  enum Tile {A,B,C,D,E,F,G,H,BLANK};

  // a position is a 0 <= row < 3 and 0 <= column < 3
  struct Position{
    int row;
    int col;
  };

  // actions are specified as moving the blank tile
  enum Action {LEFT, RIGHT, UP, DOWN};

  Puzzle() = default;

  void setLabel(Position p, Tile tile);
  
  Tile getLabel(Position p) const;

  std::pair<bool, Puzzle> apply(Action a) const;

  bool operator==(const Puzzle& rhs) const;

  // hash the 8x8 board to an int
  uint64_t hash() const;

  // a simple heuristic, city-block distance
  int heuristic(const Puzzle & goal) const;

  bool fromString(const std::string & str);

  std::string toString();
  
private:

  Tile board[9] = {A,B,C,D,E,F,G,H,BLANK};
  Position blankpos = {2,2};

  void swap(Position a, Position b);
};



#endif
