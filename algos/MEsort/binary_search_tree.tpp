#include "binary_search_tree.hpp"

template <typename KeyType, typename ItemType>
BinarySearchTree<KeyType, ItemType>::BinarySearchTree()
{
    root = 0;
}

template <typename KeyType, typename ItemType>
BinarySearchTree<KeyType, ItemType>::BinarySearchTree(
    const BinarySearchTree<KeyType, ItemType>& rhs)
{
    root = 0;
    clone(rhs.root);
}

// this is an alternative implementationusing a stack to simulate the recursion
template <typename KeyType, typename ItemType>
void BinarySearchTree<KeyType, ItemType>::clone(Node<KeyType, ItemType>* rhs)
{
    Node<KeyType, ItemType>** lhs = &root;

    std::stack<stackvar<KeyType, ItemType>> s;

    stackvar<KeyType, ItemType> rootvar;
    rootvar.rhs = rhs;
    rootvar.lhs = lhs;
    s.push(rootvar);

    while (!s.empty()) {
        stackvar<KeyType, ItemType> currentvar = s.top();
        s.pop();

        Node<KeyType, ItemType>* curr_rhs = currentvar.rhs;
        Node<KeyType, ItemType>** curr_lhs = currentvar.lhs;

        if (curr_rhs == 0)
            continue;

        // allocate new node and copy contents
        Node<KeyType, ItemType>* temp = new Node<KeyType, ItemType>;
        temp->key = curr_rhs->key;
        temp->data = curr_rhs->data;
        temp->left = 0;
        temp->right = 0;
        *curr_lhs = temp;

        // push left subtree
        currentvar.rhs = curr_rhs->left;
        currentvar.lhs = &((*curr_lhs)->left);
        s.push(currentvar);

        // push right subtree
        currentvar.rhs = curr_rhs->right;
        currentvar.lhs = &((*curr_lhs)->right);
        s.push(currentvar);
    }
}

template <typename KeyType, typename ItemType>
void BinarySearchTree<KeyType, ItemType>::destroy()
{
    std::stack<Node<KeyType, ItemType>*> s;
    s.push(root);

    while (!s.empty()) {
        Node<KeyType, ItemType>* curr = s.top();
        s.pop();

        if (curr != 0) {
            s.push(curr->left);
            s.push(curr->right);
            delete curr;
        }
    }
    root = 0;
}

template <typename KeyType, typename ItemType>
BinarySearchTree<KeyType, ItemType>& BinarySearchTree<KeyType, ItemType>::
operator=(const BinarySearchTree<KeyType, ItemType>& rhs)
{
    if (&rhs == this)
        return *this;

    destroy();

    root = 0;
    clone(rhs.root);

    return *this;
}

template <typename KeyType, typename ItemType>
BinarySearchTree<KeyType, ItemType>::~BinarySearchTree()
{
    destroy();
}

template <typename KeyType, typename ItemType>
bool BinarySearchTree<KeyType, ItemType>::insert(
    const KeyType& key, const ItemType& item)
{
    Node<KeyType, ItemType>* newnode = new Node<KeyType, ItemType>;
    newnode->key = key;
    newnode->data = item;
    newnode->left = 0;
    newnode->right = 0;
       // std::cout<<"b";
        //std::cout<<curr->data;
    if( isEmpty() )
    {
        root = newnode;
        return true;
    }

    Node<KeyType, ItemType>* curr = new Node<KeyType, ItemType>;
    Node<KeyType, ItemType>* curr_parent = new Node<KeyType, ItemType>;
    search(key, curr, curr_parent); //this returns curr as the new node where it would split from


    if( curr == 0 )
    {       std::cout<<"ERR: INSERT: SOMEHOW SKIPPED ROOT CHK"<<std::endl;
        return false;
    }

    if( curr->key > key )
    {       //std::cout<<"L\t";
        curr->left  = newnode; }
    else if( curr->key < key )
    {       //std::cout<<"R\t";
        curr->right = newnode; }
    else
    {       std::cout<<"ERR: INSERT: DUPLICATE\t"<<key<<std::endl;
        return false;
    }

    //std::cout<<"input: "<<key<<", "<<item<<"\tcurr: "<<curr->key<<", "<<curr->data<<std::endl;
    return true;
}

template <typename KeyType, typename ItemType>
bool BinarySearchTree<KeyType, ItemType>::isEmpty()
{
    return (root == 0);
}

template <typename KeyType, typename ItemType>
bool BinarySearchTree<KeyType, ItemType>::retrieve(
    const KeyType& key, ItemType& item)
{
    Node<KeyType, ItemType>* curr;
    Node<KeyType, ItemType>* curr_parent;
    search(key, curr, curr_parent);

    /*
    if( !isEmpty() )
    {   std::cout<<"root: "<<root->data<<std::endl;
        std::cout<<"curr: "<<curr->data<<std::endl; }
    */

    if (curr == 0)
        return false;

    if (curr->key == key) {
        item = curr->data;
        return true;
    }

    return false;
}

template <typename KeyType, typename ItemType>
bool BinarySearchTree<KeyType, ItemType>::remove(KeyType key)
{
    //[x]  check if empty
    //[x]  if not search for key using search
    //[x]  if none: delete
    //[x]  if one kid: replace with it
    //[x]  if two


    //check if empty
    if ( isEmpty() )
    { return false; }// empty tree

    //if not search for key using search
    Node<KeyType, ItemType>* curr = new Node<KeyType, ItemType>;
    Node<KeyType, ItemType>* curr_parent = new Node<KeyType, ItemType>;
    search(key, curr, curr_parent); //this time it returns curr as the node with matching key
            //std::cout<<"input: "<<key<<"\tcurr: "<<curr->key<<", "<<curr->data<<std::endl;

    bool rootT = false;
    if(curr->key == root->key)
    { rootT = true; }

    //check if exists
    if(curr->key != key)
    {   std::cout<<"ERR: REMOVE: NOT EXIST\t"<<std::endl;
        return false; }

    //check if roots=0
    if(curr->left == 0 && curr->right == 0)
    {  
                //if(!rootT){std::cout<<"0  ROOT:\tcurr: "<<curr->key<<", "<<curr->data<<"\tprnt: "<<curr_parent->key<<", "<<curr_parent->data<<std::endl;}
                //else     {std::cout<<"0  ROOT:\tcurr: "<<curr->key<<", "<<curr->data<<"\tprnt: EMPTY"<<std::endl;}

        //check with curr_parent
        if( rootT )  //root
        {
            root = 0;
            return true;
        }
        
        else if( curr_parent != 0 && curr_parent->left == curr )
        { curr_parent->left  = 0; }
        
        else if( curr_parent != 0 && curr_parent->right == curr )
        { curr_parent->right = 0; }
        
        else
        {   std::cout<<"ERR: REMOVE: ROOT 0 COLLISION\t"<<std::endl;
        return false; }

        delete curr;
        return true;
    }

    //check if roots=1
    else if( (curr->left == 0 && curr->right != 0) || (curr->left != 0 && curr->right == 0) )
    {
                //if(!rootT){std::cout<<"1  ROOT:\tcurr: "<<curr->key<<", "<<curr->data<<"\tprnt: "<<curr_parent->key<<", "<<curr_parent->data;}
                //else     {std::cout<<"1  ROOT:\tcurr: "<<curr->key<<", "<<curr->data<<"\tprnt: EMPTY";}

        Node<KeyType, ItemType>* tempr;

        if(curr->left != 0)       //left
        {       //std::cout<<"\tL: "<<curr->left->key<<", "<<curr->left->data<<std::endl;

            if(rootT)
            {   tempr = root;
                root = root->left;
                delete tempr; }
            else
            {
                if(curr_parent->left == curr)
                { curr_parent->left = curr->right; }
                else if(curr_parent->right == curr)
                { curr_parent->right = curr->right; }
                else
                {   std::cout<<"ERR: REMOVE: ROOT 0 COLLISION\t"<<std::endl;
                return false; }

                delete curr;
            }
            return true;
        }
        else if(curr->right != 0)  //right
        {       //std::cout<<"\tR: "<<curr->right->key<<", "<<curr->right->data<<std::endl;

            if(rootT)
            {   tempr = root;
                root = root->right;
                delete tempr; }
            else
            {
                if(curr_parent->left == curr)
                { curr_parent->left = curr->right; }
                else if(curr_parent->right == curr)
                { curr_parent->right = curr->right; }
                else
                {   std::cout<<"ERR: REMOVE: ROOT 0 COLLISION\t"<<std::endl;
                return false; }

                delete curr;
            }
            return true;
        }
        else
        {   std::cout<<"\nERR: REMOVE: ROOT 1 COLLISION\t"<<std::endl;
        return false; }
    }

    //check if roots=2
    else if(curr->left != 0 && curr->right != 0)
    {
            //if(!rootT){std::cout<<"2  ROOT:\tcurr: "<<curr->key<<", "<<curr->data<<"\tprnt: "<<curr_parent->key<<", "<<curr_parent->data<<std::endl;}
            //else     {std::cout<<"2  ROOT:\tcurr: "<<curr->key<<", "<<curr->data<<"\tprnt: EMPTY"<<std::endl;}

        Node<KeyType, ItemType>* minnode;
        inorder(curr, minnode, curr_parent);    //def min as smallest value to right of curr

        KeyType tkey = minnode->key;    ItemType tdata = minnode->data;
        remove(minnode->key);
        
        curr->key  = tkey;
        curr->data = tdata;
        return true;
    }

    return false; // default should never get here
}

template <typename KeyType, typename ItemType>
void BinarySearchTree<KeyType, ItemType>::inorder(Node<KeyType, ItemType>* curr,
    Node<KeyType, ItemType>*& in, Node<KeyType, ItemType>*& parent)
{
    while(curr->left != 0)
    { curr = curr->left; }
    in = curr;
}

int depth = 0;

template <typename KeyType, typename ItemType>
void BinarySearchTree<KeyType, ItemType>::search(KeyType key,
    Node<KeyType, ItemType>*& curr, Node<KeyType, ItemType>*& parent)
{
    curr = root;
    parent = 0;

    if (isEmpty())
        return;
    

    while (true) {
        if (key == curr->key) {
            break;
        } else if (key < curr->key) {
            if (curr->left != 0) {
                parent = curr;
                curr = curr->left;
            } else {
                break;
            }
        } else {
            if (curr->right != 0) {
                parent = curr;
                curr = curr->right;
            } else {
                break;
            }
        }
    }
}

template<typename KeyType, typename ItemType>
void BinarySearchTree<KeyType, ItemType>::treeSort(ItemType arr[], int size){
// TODO: use the tree to sort the array items
    int newsize=0;
    for(int i=0;i<size;i++)
    { if( insert( arr[i] , arr[i] ) ){newsize++;} }//auto deals with duplicates
            //std::cout<<newsize<<std::endl;

// TODO: overwrite input array values with sorted values
    //unset array
    for(int i=0;i<size;i++)
    { arr[i] = NULL; }

    //re-set array with new values
    Node<KeyType, ItemType>* least = new Node<KeyType, ItemType>;
    Node<KeyType, ItemType>* parent = new Node<KeyType, ItemType>;
    for(int i=0;i<newsize;i++)
    {
        inorder(root,least,parent);
        arr[i] = least->data;
        remove(least->key);
    }

}

template <typename KeyType, typename ItemType>
ItemType BinarySearchTree<KeyType, ItemType>::lowest()
{
    Node<KeyType, ItemType>* least = new Node<KeyType, ItemType>;
    Node<KeyType, ItemType>* parent = new Node<KeyType, ItemType>;
    inorder(root,least,parent);
    ItemType ret = least->data;

            //std::cout<<"low: "<<ret       <<"\n";
            //std::cout<<"key: "<<least->key<<"\n";

    remove(least->key);
    return ret;
}