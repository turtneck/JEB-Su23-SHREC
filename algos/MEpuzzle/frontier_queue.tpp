#include "frontier_queue.hpp"
#include <iostream>
#include <cstdlib>

//min heap

// TODO implement the member functions here

//find lowest f-cost, remove then return it
template <typename T>
State<T> frontier_queue<T>::pop() {
  
  if(queue.empty())
  { std::exit(EXIT_FAILURE);}
  
      //if(queue.size() == 1)
      //{ std::cout<<"pop, F: "<<(queue.at(0)).getFCost()<<std::endl; }


  T           v = (queue.front()) .getValue();
  std::size_t c = (queue.front()) .getPathCost();
  std::size_t h = (queue.front()) .getFCost() - c;
  //State<T> temp( v,c,h );

  int j=0;
  std::size_t lowest = (queue.front()).getFCost();
  for(int i=0;i<queue.size();i++)
  {
        //std::size_t a = (queue.at(i)).getFCost();
        //std::cout<<"pop, F: "<<a<<std::endl;
    
    if( (queue.at(i)).getFCost() < lowest )
    {
      j=i;
      lowest = (queue.at(i)).getFCost();
      v = (queue.at(i)).getValue();
      c = (queue.at(i)).getPathCost();
      h = (queue.at(i)).getFCost() - c;
    }
  }

  queue.erase(queue.begin()+j);

  State<T> temp( v,c,h );
  return temp;
}

//value p, cost, and heuristic
//min sorted (with value?)
template <typename T>
void frontier_queue<T>::push(const T &p, std::size_t cost, std::size_t heur) {
      //if(queue.empty())
      //{ std::cout<<"push, V: "<<p<<std::endl; }
  /*//misconception that i needed to sort it
  int j=0;
  //State<T> temp(p,cost,heur);
  for(int i=0;i<queue.size();i++)
  {
        //T a = (queue.at(i)).getValue();
        //std::cout<<"push, V: "<<a<<std::endl;
    j=i;
    if( (queue.at(i)).getValue() > p )
    { break; }
    
  }*/

  State<T> newstate(p,cost,heur);

  //queue.insert(queue.begin()+j, newstate);

  queue.push_back(newstate);

}

template <typename T>
bool frontier_queue<T>::empty() {

  return queue.empty();
}

template <typename T> 
bool frontier_queue<T>::contains(const T &p) {

  for(int i=0;i<queue.size();i++)
  {
    if( (queue.at(i)).getValue() == p )
    { return true; }
  }
  return false;

}

//replace path cost of value state if cost is lower
template <typename T>
void frontier_queue<T>::replaceif(const T &p, std::size_t cost) {
  if(queue.empty())
  { std::cout<<"\n=======================\nERR: REPLACEIF: EMPTY\n=======================\n";
    //exit(EXIT_FAILURE);
    return; }
  
  for(int i=0;i<queue.size();i++)
  {
    if( (queue.at(i)).getValue() == p )
    { if( queue.at(i).getPathCost() > cost)
      { queue.at(i).updatePathCost(cost);} }
  }
}


