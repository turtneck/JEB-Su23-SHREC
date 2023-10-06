#include "Database.hpp"
#include "list.hpp"

template <typename T>
Database<T>::Database() {}

template <typename T>
bool Database<T>::isEmpty() { return dict.isEmpty() + K1.isEmpty() + K2.isEmpty() + t1.empty() + t2.empty(); }

template <typename T>
std::size_t Database<T>::getNumberOfEntries() { return dict.getLength(); }

template <typename T>
bool Database<T>::add(std::string key1, std::string key2, const T& item) {
            //std::cout<<"L: "<<dict.getLength()+1<<std::endl;
    
    bool
    ret  = dict.insert(dict.getLength()+1,item);   //add to end
//std::cout<<dict.getLength();
    ret *= K1.insert(key1,dict.getLength());
//std::cout<<dict.getLength();
    ret *= K2.insert(key2,dict.getLength());

    t1.push_back(key1);
    t2.push_back(key2);
        
            //std::cout<<"K1: "<<K1.tempfunc1()<<", "<<K1.tempfunc2()<<std::endl;
            //std::cout<<"K2: "<<K2.tempfunc1()<<", "<<K2.tempfunc2()<<std::endl;

    return ret;
}

template <typename T>
bool Database<T>::remove(std::string key) {
    int pos;
    bool ret=true;

    if(K1.retrieve(key,pos))
    {
                //std::cout<<"K1pos: "<<pos<<std::endl;
                //std::cout<<t2.at(pos-1)<<std::endl;
        ret *= K1.remove(key);
        ret *= K2.remove( t2.at(pos-1) );
    }
    else if(K2.retrieve(key,pos))
    {
                //std::cout<<"K2pos: "<<pos<<std::endl;
                //std::cout<<t1.at(pos-1)<<std::endl;

        ret *= K2.remove(key);
        ret *= K1.remove( t1.at(pos-1) );
    }
    else{return false;}

    dict.remove(pos);
    t1.erase(t1.begin()+pos-1); t2.erase(t2.begin()+pos-1);
    return ret;
}

template <typename T>
void Database<T>::clear() {
    dict.clear();
    K1.destroy();
    K2.destroy();
}

template <typename T>
T Database<T>::getValue(std::string key) {
    int pos=dict.getLength()+5;

    K1.retrieve(key,pos);
    if(pos>dict.getLength() || pos<=0)
    { K2.retrieve(key,pos); }   //wierd error with retrieve

            //std::cout<<"pos: "<<pos<<std::endl;
    
    return dict.getEntry(pos);
    
}

template <typename T>
bool Database<T>::contains(std::string key) {
    int pos;
    bool
    ret  = K1.retrieve(key,pos);
    ret += K2.retrieve(key,pos);
    return ret;
}

// return all entries in search key order: 
// sort by key1 if keyIndex==1, key2 if keyIndex==2
// key1 and key2 should be consistent with the add method parameters
template <typename T>
std::vector<T> Database<T>::getAllEntries(int keyIndex) {
    
    std::vector<T> keylist;
    BinarySearchTree<std::string,int> K1_temp = K1;
    BinarySearchTree<std::string,int> K2_temp = K2;

    int cout1; //std::string cout2;

    if(keyIndex == 1)   //key1
    {   for(int i=0;i<dict.getLength();i++)
        {
            cout1 = K1.lowest();
                    //std::cout<<"p: "<< cout1 <<"\n";//", "<< cout2 <<std::endl;

            keylist.push_back(  dict.getEntry( cout1 )  );
        }  
    }

    else if(keyIndex == 2)   //key2
    {   for(int i=0;i<dict.getLength();i++)
        {
            cout1 = K2.lowest();
                    //std::cout<<"p: "<< cout1 <<"\n";//", "<< cout2 <<std::endl;
            keylist.push_back(  dict.getEntry( cout1 )  );
        } 
    }

    else { exit(EXIT_FAILURE); }

    K1 = K1_temp;
    K2 = K2_temp;   //restore binary trees

    return keylist;
}