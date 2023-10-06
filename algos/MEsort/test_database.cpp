#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_COLOUR_NONE
#include "catch.hpp"

#include "Database.hpp"
#include "list.hpp"
#include "binary_search_tree.hpp"


TEST_CASE("list","[test list]"){
    List<char> lst;
    REQUIRE(lst.isEmpty());

    lst.insert(1,'a');
    REQUIRE(!lst.isEmpty());

    lst.remove(1);
    REQUIRE(lst.isEmpty());

    //length
    REQUIRE(lst.getLength() == 0);

    lst.insert(1,'b');
    REQUIRE(lst.getLength() == 1);

    lst.insert(2,'a');
    REQUIRE(lst.getLength() == 2);

    lst.remove(1);
    REQUIRE(lst.getLength() == 1);

    lst.remove(1);
    REQUIRE(lst.getLength() == 0);


    //insert/remove/getentry
    char c = 'a';
    for (int i = 0; i < 26; ++i) {
        REQUIRE(lst.insert(i+1,c+i));
    }

    REQUIRE(lst.getLength() == 26);

    for (int i = 0; i < 26; ++i) {
        REQUIRE(lst.getEntry(i+1) == 'a' + i);
    }

    lst.clear();
    REQUIRE(lst.getLength() == 0);

    c = 'a';
    for (int i = 0; i < 26; ++i) {
        REQUIRE(lst.insert(i+1,c+i));
    }
    REQUIRE(lst.getLength() == 26);

//readout(lst);

    lst.remove(6);

//std::cout<<"\n\n===\n";
//readout(lst);

    REQUIRE(lst.getLength() == 25);
    REQUIRE(lst.getEntry(6) == 'g');

    lst.remove(12);
    REQUIRE(lst.getLength() == 24);
    REQUIRE(lst.getEntry(12) == 'n');

    lst.remove(1);
    REQUIRE(lst.getLength() == 23);
    REQUIRE(lst.getEntry(1) == 'b');

    lst.remove(23);
    REQUIRE(lst.getLength() == 22);
    REQUIRE(lst.getEntry(22) == 'y');

    lst.clear();

    //copy
    c = 'a';
    for (int i = 0; i < 26; ++i) {
        REQUIRE(lst.insert(i+1,c+i));
    }
    REQUIRE(lst.getLength() == 26);

    List<char> lst_copy = lst;

    REQUIRE(lst_copy.getLength() == 26);

    for (int i = 0; i < 26; ++i) {
        REQUIRE(lst_copy.getEntry(i+1) == lst.getEntry(i+1));
    }

    List<char> lst1;
    List<char> lst_copy1 = lst1;

    REQUIRE(lst1.getLength() == 0);
    REQUIRE(lst_copy1.getLength() == 0);

    lst.clear();

    /*
    CHECK_THROWS_AS(lst.remove(1), std::range_error);
    CHECK_THROWS_AS(lst.getEntry(1), std::range_error);

    lst.insert(1,'a');
    lst.insert(2,'b');
    lst.insert(3,'c');
    lst.insert(4,'d');

    
    CHECK_THROWS_AS(lst.remove(5), std::range_error);
    CHECK_THROWS_AS(lst.remove(6), std::range_error);
    CHECK_THROWS_AS(lst.getEntry(5), std::range_error);
    CHECK_THROWS_AS(lst.getEntry(6), std::range_error);

    CHECK_THROWS_AS(lst.remove( 0), std::range_error);
    CHECK_THROWS_AS(lst.remove(-1), std::range_error);
    CHECK_THROWS_AS(lst.getEntry( 0), std::range_error);
    CHECK_THROWS_AS(lst.getEntry(-1), std::range_error);
    */
}


typedef BinarySearchTree<int, int> TreeType;
TEST_CASE("BST","[test BST]"){
    TreeType bst1;
    REQUIRE(bst1.isEmpty());

    bst1.insert(10, 10);
    REQUIRE(!bst1.isEmpty());
    bst1.insert(12, 12);
    REQUIRE(!bst1.insert(12, 12));
    
    TreeType bst2;
    bst2.insert(10, 10);
    bst2.insert(5, 5);
    bst2.insert(15, 15);
    bst2.insert(12, 12);
    bst2.insert(18, 18);

    int item;
    REQUIRE(bst2.retrieve(18, item));
    REQUIRE(bst2.retrieve(12, item));
    REQUIRE(bst2.retrieve(15, item));
    REQUIRE(bst2.retrieve(5 , item));
    REQUIRE(bst2.retrieve(10, item));

    bst2.remove(12);
    REQUIRE(!bst2.retrieve(12, item));
    bst2.remove(18);
    REQUIRE(!bst2.retrieve(18, item));
    bst2.remove(5);
    REQUIRE(!bst2.retrieve(5, item));
    bst2.remove(10);
    REQUIRE(!bst2.retrieve(10, item));
    bst2.remove(15);
    REQUIRE(!bst2.retrieve(15, item));

    REQUIRE(bst2.isEmpty());

    REQUIRE(bst2.insert(10, 10));
    bst2.insert(5, 5);
    bst2.insert(15, 15);
    bst2.insert(12, 12);
    bst2.insert(18, 18);

    bst2.remove(15);
    REQUIRE(!bst2.retrieve(15, item));
    REQUIRE( bst2.retrieve(12, item));
    REQUIRE( bst2.retrieve( 5, item));
    REQUIRE( bst2.retrieve(10, item));
    REQUIRE( bst2.retrieve(18, item));

    //test copy
    TreeType bsta1;

    bsta1.insert(50, 50);
    bsta1.insert(0, 0);
    bsta1.insert(100, 100);
    bsta1.insert(25, 25);
    bsta1.insert(75, 75);

    TreeType bsta2;

    bsta2 = bsta1;

    REQUIRE(!bsta2.isEmpty());

    REQUIRE( bsta2.retrieve(100, item));
    REQUIRE( bsta2.retrieve(75,  item));
    REQUIRE( bsta2.retrieve(50,  item));
    REQUIRE( bsta2.retrieve(25,  item));
    REQUIRE(!bsta2.retrieve(51,  item));

    //treesort
    std::cout<<"\nTREESORT\n";
    TreeType tree;
    int arr[] = {6,6,7,6,9,1,9,0,0,1,4,5,1};
    int size = sizeof(arr) / sizeof(int);

    std::cout<<"Input: ";
    for(int i=0;i<size-1;i++)
    {std::cout<<arr[i]<<", ";}
    std::cout<<arr[size-1]<<std::endl;

    tree.treeSort(arr,size);

    std::cout<<"Output: ";
    for(int i=0;i<size-1;i++)
    {std::cout<<arr[i]<<", ";}
    std::cout<<arr[size-1]<<std::endl;
}


struct Entry {
    std::string title;
    std::string author;
    int pubYear;
};


TEST_CASE("Test Construction", "[construction]") {
    std::cout<<"=============Test Construction=============\n";

    Database<std::string> testdb;
    REQUIRE(testdb.isEmpty());
}


TEST_CASE("Test Add", "[add]") {
    std::cout<<"=============Test Add=============\n";
    
    Database<std::string> testdb;
    std::string e1 = "entry";

    testdb.add("key1", "key2", e1);
    REQUIRE(!testdb.isEmpty());
    REQUIRE(testdb.contains("key1"));
    REQUIRE(testdb.contains("key2"));
}


TEST_CASE("Test Duplicate Add", "[duplicate add]") {
    std::cout<<"=============Test Dup Add=============\n";

    Database<std::string> testdb;
    std::string e1 = "entry";
    
    testdb.add("key1", "key2", e1);
    REQUIRE(!testdb.add("key1", "key2", e1));
}


TEST_CASE("Test Retrieve", "[retrieve]") {
    std::cout<<"=============Test Retrieve=============\n";

    Database<std::string> testdb;
    std::string e1 = "entry";

    testdb.add("key1", "key2", e1);
    
    REQUIRE(testdb.getValue("key1") == e1);
    REQUIRE(testdb.getValue("key2") == e1);
}


TEST_CASE("Test Add 2", "[add 2]") {
    std::cout<<"=============Test Add 2=============\n";

    Database<std::string> testdb;
    std::string e1 = "entry1";
    std::string e2 = "entry2";
    
    testdb.add("key1a", "key1b", e1);
    testdb.add("key2a", "key2b", e2);

    REQUIRE(testdb.getNumberOfEntries() == 2);

    REQUIRE(testdb.getValue("key1a") == e1);
    REQUIRE(testdb.getValue("key1b") == e1);
    REQUIRE(testdb.getValue("key2a") == e2);
    REQUIRE(testdb.getValue("key2b") == e2);
}


TEST_CASE("Test Remove", "[remove]") {
    std::cout<<"=============Test Remove=============\n";

    Database<std::string> testdb;
    std::string e1 = "entry";

    testdb.add("key1", "key2", e1);

    testdb.remove("key1");
    REQUIRE(!testdb.contains("key1"));
    REQUIRE(!testdb.contains("key2"));
    REQUIRE( testdb.isEmpty() );

    testdb.add("key1", "key2", e1);

    testdb.remove("key2");

    REQUIRE(!testdb.contains("key1"));
    REQUIRE(!testdb.contains("key2"));

    REQUIRE(testdb.isEmpty());
}


TEST_CASE("Test Copy Construct", "[copy]") {
    std::cout<<"=============Test Copy Construct=============\n";

    Database<std::string> testdb;
    std::string e1 = "entry";

    testdb.add("key1", "key2", e1);

    Database<std::string> testdb_copy(testdb);

    REQUIRE(testdb_copy.getValue("key1") == e1);
    REQUIRE(testdb_copy.getValue("key2") == e1); 
}


TEST_CASE("Test Copy Assign", "[copy assign]") {
    std::cout<<"=============Test Copy Assign=============\n";

    Database<std::string> testdb;
    std::string e1 = "entry";

    testdb.add("key1", "key2", e1);

    Database<std::string> testdb_copy;

    testdb_copy = testdb;

    REQUIRE(testdb_copy.getValue("key1") == e1);
    REQUIRE(testdb_copy.getValue("key2") == e1); 
}

TEST_CASE("Test Entry Types", "[entry type]") {
    std::cout<<"=============Test Entry Types=============\n";

    Database<Entry> testdb;

    Entry e1;
    e1.title = "The Winds of Winter";
    e1.author = "George R. R. Martin";
    e1.pubYear = 2031;

    std::string isbn1 = "000-0-00-000000-1";
    std::string catalog_id1 = "0000001";

    testdb.add(isbn1, catalog_id1, e1);

    Entry e2;
    e2.title = "A Dream of Spring";
    e2.author = "George R. R. Martin";
    e2.pubYear = 2032;

    std::string isbn2 = "000-0-00-000000-2";
    std::string catalog_id2 = "0000002";

    testdb.add(isbn2, catalog_id2, e2);
    
}


TEST_CASE("Test getAllEntries string", "[getAllEntries]") {
    std::cout<<"=============Test getAllEntries - String =============\n";

    Database<std::string> testdb;

    std::vector<std::string> inputvec;
    for (int i=0;i<26;i++)
    {
        if(i>9){ inputvec .push_back( "Title: "   + std::to_string(i));}
        else   { inputvec .push_back( "Title: 0"  + std::to_string(i));}
    }
    
    std::string isbn;
    std::string catalog_id;
    for(int i=0;i<26;i++)
    {

        if(i>9){ isbn = "000-0-00-000000-"  + std::to_string(i);}
        else   { isbn = "000-0-00-000000-0" + std::to_string(i);}

        if(25-i>9){ catalog_id = "00000"  + std::to_string(25-i);}
        else   { catalog_id = "000000" + std::to_string(25-i);}

                //std::cout<<isbn<<", "<<catalog_id<<", "<<inputvec.at(i)<<std::endl;
        
        testdb.add(isbn, catalog_id, inputvec.at(i));
    }


    std::vector<std::string> vec = testdb.getAllEntries(1);
    //print vec1
    std::cout<<"VEC1: \n";
    for(int i=0;i<26;i++)
    {   std::cout<<"output: "<<vec.at(i)<<std::endl;  }
    
    
    //print vec2
    vec = testdb.getAllEntries(2);
    std::cout<<"\n\nVEC2: \n";
    for(int i=0;i<26;i++)
    {   std::cout<<"output: "<<vec.at(i)<<std::endl;  }

}


TEST_CASE("Test getAllEntries Entry", "[getAllEntries]") {
    std::cout<<"=============Test getAllEntries - Entry =============\n";

    Database<Entry> testdb;

    std::vector<std::string> titlev, authorv;
    for (int i=0;i<26;i++)  { titlev .push_back( "Title: "  + std::to_string(i) ); }
    for (int i=25;i>=0;i--) { authorv.push_back( "Author: " + std::to_string(i) ); }

            //for (int i=0;i<26;i++) { std::cout<<"T:\t"<<titlev .at(i)<<std::endl; }
            //for (int i=0;i<26;i++) { std::cout<<"A:\t"<<authorv.at(i)<<std::endl; }

    Entry e1;
    std::string isbn;
    std::string catalog_id;
    for(int i=0;i<26;i++)
    {
        e1.title  = titlev .at(i);
        e1.author = authorv.at(i);
        e1.pubYear= i;

        if(i>9){ isbn = "000-0-00-000000-"  + std::to_string(i);}
        else   { isbn = "000-0-00-000000-0" + std::to_string(i);}

        if(i>9){ catalog_id = "00000"  + std::to_string(i);}
        else   { catalog_id = "000000" + std::to_string(i);}

                //std::cout<<isbn<<", "<<catalog_id<<std::endl;
        testdb.add(isbn,catalog_id,e1);
    }


    std::vector<Entry> vec = testdb.getAllEntries(1);
    //print vec1
    std::cout<<"VEC1: \n";
    for(int i=0;i<26;i++)
    {   e1 = vec.at(i);
        std::cout<<e1.title<<",\t"<<e1.author<<",\t"<<e1.pubYear<<std::endl;  }
    
    
    //print vec2
    vec = testdb.getAllEntries(2);
    std::cout<<"\n\nVEC2: \n";
    for(int i=0;i<26;i++)
    {   e1 = vec.at(i);
        std::cout<<e1.title<<",\t"<<e1.author<<",\t"<<e1.pubYear<<std::endl;  }

}



