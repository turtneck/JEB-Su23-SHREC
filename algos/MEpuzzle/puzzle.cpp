#include "puzzle.hpp"

void Puzzle::setLabel(Position p, Tile tile) {
  if ((p.row < 0) || (p.row > 2) || (p.col < 0) || (p.col > 2)) {
    throw std::range_error("invlid row/column in Puzzle::setLabel");
  }
  board[3 * p.row + p.col] = tile;
}

Puzzle::Tile Puzzle::getLabel(Position p) const {
  if ((p.row < 0) || (p.row > 2) || (p.col < 0) || (p.col > 2)) {
    throw std::range_error("invlid row/column in Puzzle::getLabel");
  }
  return board[3 * p.row + p.col];
}

std::pair<bool, Puzzle> Puzzle::apply(Action a) const {
  Puzzle temp = *this;

  bool valid = false;

  if ((a == LEFT) && (blankpos.col > 0)) {
    temp.blankpos.col -= 1;
    temp.swap(blankpos, temp.blankpos);
    valid = true;
  } else if ((a == RIGHT) && (blankpos.col < 2)) {
    temp.blankpos.col += 1;
    temp.swap(blankpos, temp.blankpos);
    valid = true;
  } else if ((a == UP) && (blankpos.row > 0)) {
    temp.blankpos.row -= 1;
    temp.swap(blankpos, temp.blankpos);
    valid = true;
  } else if ((a == DOWN) && (blankpos.row < 2)) {
    temp.blankpos.row += 1;
    temp.swap(blankpos, temp.blankpos);
    valid = true;
  }

  return std::make_pair(valid, temp);
}

bool Puzzle::operator==(const Puzzle &rhs) const {
  return (heuristic(rhs) == 0);
}

uint64_t Puzzle::hash() const {
  uint64_t h = board[0];
  for (int i = 1; i < 9; ++i) {
    h = (h << 4) + board[i];
  }
  return h;
}

// a simple heuristic, city-block distance
int Puzzle::heuristic(const Puzzle &goal) const {
  int count = 0;

  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 9; ++j) {
      if (board[i] == goal.board[j]) {
        int ic = i % 3;
        int ir = i / 3;
        int jc = j % 3;
        int jr = j / 3;
        int cd = (ic > jc) ? (ic - jc) : (jc - ic);
        int rd = (ir > jr) ? (ir - jr) : (jr - ir);
        count += cd + rd;
      }
    }
  }
  return count;
}

bool Puzzle::fromString(const std::string &str) {

  if (str.size() != 9) {
    return false;
  }

  for (auto c : str) {
    if ((c < '0') || (c > '8')) {
      return false;
    }
  }

  int i = 0;
  int count[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (auto c : str) {
    int temp = c - '0';
    count[temp] += 1;
    board[i] = static_cast<Tile>(temp);
    if (board[i] == BLANK) {
      blankpos.row = i / 3;
      blankpos.col = i % 3;
    }
    i++;
  }

  for (i = 0; i < 9; ++i) {
    if (count[i] != 1)
      return false;
  }

  return true;
}

std::string Puzzle::toString() {

  std::ostringstream oss;
  for (int i = 0; i < 9; ++i) {
    oss << board[i];
  }
  return oss.str();
}

void Puzzle::swap(Position a, Position b){
  std::swap(board[3*a.row + a.col], board[3*b.row + b.col]);
}
