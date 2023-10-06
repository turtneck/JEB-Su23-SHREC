#include "list.hpp"

template <typename T>
List<T>::List()
{
  first = nullptr;
}

template <typename T>
List<T>::~List()
{
  clear();
}

template <typename T>
List<T>::List(const List<T>& x)
{
  //base = x.base;

  first = NULL;
  
  std::size_t size = x.getLength();

  for(std::size_t i=1;i<size+1;i++) { insert(i, x.getEntry(i) ); }
}

template <typename T>
List<T>& List<T>::operator=(const List<T>& x)
{
  //base = x.base;
  //return *this;

  clear();

  std::size_t size = x.getLength();
  for(std::size_t i=1;i<size+1;i++) { insert(i, x.getEntry(i) ); }
  return *this;
}

template <typename T>
bool List<T>::isEmpty() const
{
  //return base.empty();

  return first == nullptr;
}

template <typename T>
std::size_t List<T>::getLength() const
{
  //return base.size();

  if( isEmpty() ) { return 0; }
  
  Node<T>* temp = first;
  std::size_t leng=0;

  while(temp != nullptr)
  { temp = temp->next;
    leng++; }

  return leng;
}

template <typename T>
bool List<T>::insert(std::size_t position, const T& item)
{
  //base.insert(base.begin()+position-1,item);
  //return true;

  Node<T>* newnode = new Node<T>;
  newnode->data = item;

    //checks
  if( isEmpty() )
  { first = newnode;
    return true; }
  
  std::size_t size = getLength();

  if( size+1 < position)
  {return false;}
    //pass checks

    //exceptions
  if(position == 1)
  {
    newnode->next = first;
    first = newnode;
    return true;
  }

    //actual work
  Node<T>* temp = first;
  Node<T>* hold = nullptr;

  std::size_t num = position;
  while(num > 1)
  {
    hold = temp;
    temp = temp->next;
    num--;
  }
  //temp = temp->getNext();  //temp = pnt front of target
  //hold = hold->getNext(); //hold = target

  hold->next = newnode;
  newnode->next = temp;

  return true;
}


template <typename T>
bool List<T>::remove(std::size_t position)
{
  //base.erase(base.begin()+position-1);
  //return true;

    //checks
  if( isEmpty() )
  { //throw std::range_error("remove: empty");
    return false;}

  std::size_t size = getLength();

  if( size+1 < position || position < 0)
  { //throw std::range_error("remove: out of range");
    return false;}
    //pass checks

    //exceptions (delete first)
  if(position == 1)
  {
    first = first->next;
    return true;
  }

    //actual work
  Node<T>* temp = first;
  Node<T>* hold = nullptr;

  if(position == size)
  {
        //std::cout<<"kok\n";
    Node<T>* hold2 = nullptr;

    while(temp != nullptr)
    {
      hold2 = hold;
      hold = temp;
      temp = temp->next;
    }
    hold2->next = nullptr;
    return true;
  }

  std::size_t num = position;
  while(num > 1)
  {
    
    hold = temp;
    temp = temp->next;
    
    num--;
  }
  //temp = temp->next;  //temp = pnt next to want-to-delete
  //hold = hold->next;  //hold = want-to-delete

      //std::cout<<hold->data<<", "<<temp->data<<"\n";
  hold->next = temp->next;

  hold = temp;  //the want to delete is skipped over to be the next
  delete temp;  //data is erased

  return true;
}

template <typename T>
void List<T>::clear()
{
  //base.clear();

  while( !isEmpty() )
  { remove(1); }

  /*
  if( isEmpty() ) { return; }

  Node<T>* temp = first;
  Node<T>* temp1 = temp;
  while(temp != nullptr)
  {
    temp1 = temp;
    temp = nullptr;
    temp1 = temp1->next;
    temp = temp1;
  }
  */
}

template <typename T>
T List<T>::getEntry(std::size_t position) const
{
  //return base.at(position-1);

  if( isEmpty() ) { exit(EXIT_FAILURE); }

  Node<T>* temp = first;

  std::size_t num = position;
  while(num > 1)
  {
    temp = temp->next;
    num--;
  }

  return temp->data;
}

template <typename T>
void List<T>::setEntry(std::size_t position, const T& newValue)
{
  //base.at(position-1) = newValue;

  if( isEmpty() ) { return; }

  Node<T>* temp = first;

  std::size_t num = position;
  while(num > 1)
  {
    temp = temp->next;
    num--;
  }

  temp->data = newValue;
}

template <typename T>
void readout( List<T> a)
{
      //std::cout<<a.getLength()<<std::endl;
  for(int i=0;i<a.getLength();i++)
  { std::cout<<a.getEntry(i+1)<<std::endl; }
}