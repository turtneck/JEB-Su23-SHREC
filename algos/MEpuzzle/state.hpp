#ifndef STATE_HPP
#define STATE_HPP

// generic state holding a value of type T
template <typename T>
class State {
public:
  State(const T &p, std::size_t cost, std::size_t heur);

  T getValue() const;
  
  void updatePathCost(std::size_t cost);
  
  std::size_t getPathCost() const;

  std::size_t getFCost() const;

private:
  
  T value;
  std::size_t g {0};
  std::size_t h {0};
  std::size_t f {0};
};

#include "state.tpp"

#endif
