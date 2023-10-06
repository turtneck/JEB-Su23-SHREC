
#ifndef _DATABASE_H_
#define _DATABASE_H_

#include "list.hpp"
#include "binary_search_tree.hpp"
#include <iostream>
#include <vector>

template <typename T>
class Database
{
public:
  Database();

  // determine if the database is empty
  bool isEmpty();

  // return current number of entries
  std::size_t getNumberOfEntries();

  // insert entry 
  bool add(std::string key1, std::string key2, const T& item);

  // remove entry 
  bool remove(std::string key);

  // remove all entries
  void clear();

  // retrieve the value of the entry associated with the input key
  T getValue(std::string key);

  // return true if an entry exists associated with the input key
  bool contains(std::string key);

  // return all entries in search key order: 
  // sort by key1 if keyIndex==1, key2 if keyIndex==2
  // key1 and key2 should be consistent with the add method parameters
  std::vector<T> getAllEntries(int keyIndex);

private:

  List<T> dict; //list of data
  BinarySearchTree<std::string,int> K1;// = new BinarySearchTree<std::string,int>; //store key1s with info of pos of data in dict
  BinarySearchTree<std::string,int> K2;// = new BinarySearchTree<std::string,int>; //store key2s with info of pos of data in dict
  std::vector<std::string> t1;
  std::vector<std::string> t2;
};

#include "Database.tpp"
#endif // _DATABASE_H_
