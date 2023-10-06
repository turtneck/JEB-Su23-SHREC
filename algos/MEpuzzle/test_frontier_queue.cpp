#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "frontier_queue.hpp"

TEST_CASE("Simple push test", "[frontier_queue]")
{
    std::cout<<"Simple push test\n";

    frontier_queue<int> fq;
    REQUIRE(fq.empty());

    fq.push(1, 0, 1);

    REQUIRE_FALSE(fq.empty());
    REQUIRE(fq.contains(1));
}

TEST_CASE("Simple contains test", "[frontier_queue]")
{
    std::cout<<"Simple contains test\n";

    frontier_queue<int> fq;

    REQUIRE_FALSE(fq.contains(1));
    REQUIRE_FALSE(fq.contains(2));

    fq.push(1, 1, 1);
    fq.push(2, 2, 2);

    REQUIRE(fq.contains(1));
    REQUIRE(fq.contains(2));
    REQUIRE_FALSE(fq.contains(3));
}

TEST_CASE("Simple pop test", "[frontier_queue]")
{
    std::cout<<"Simple pop test\n";
    
    frontier_queue<int> fq;
    
    fq.push(1, 1, 1);

    State<int> state = fq.pop();
    REQUIRE(state.getValue() == 1);
    REQUIRE(state.getPathCost() == 1);
    REQUIRE(state.getFCost() == 2);
    REQUIRE(fq.empty());
}

TEST_CASE("Simple replaceif test", "[frontier_queue]")
{
    std::cout<<"Simple replaceif test\n";
    
    frontier_queue<int> fq;

    fq.push(1, 100, 100);

    fq.replaceif(1, 1);
    State<int> state = fq.pop();
    REQUIRE(state.getValue() == 1);
    REQUIRE(state.getPathCost() == 1);
    REQUIRE(state.getFCost() == 101);
}

TEST_CASE("Thorough push/pop test", "[frontier_queue]")
{
    std::cout<<"Thorough push/pop test\n";

    // Init random numbers
    int seed = rand() % 100;
    INFO("Seed = " << seed);
    srand(seed);

    // Fill the frontier_queue with random entries
    frontier_queue<int> fq;
    for (int i = 0; i < 5; i++)
    {
        fq.push(rand() % 1000, i, i);
    }

    std::vector<State<int>> states;
    while (!fq.empty())
    {
        State<int> state = fq.pop();
        states.push_back(state);
    }

    // Require that pop gave us the highest priority every time
    for (int i = 0; i < states.size() - 1; i++)
    {
        REQUIRE(states[i].getFCost() <= states[i+1].getFCost());
    }
    std::cout<<"success\n";
}

TEST_CASE("Thorough replaceif test", "[frontier_queue]")
{
    std::cout<<"Thorough replaceif test\n";
    
    frontier_queue<int> fq;

    // Try replaceif on empty queue
    fq.replaceif(2, 1);
    REQUIRE(fq.empty());

    // Try to replace entry that's not there
    fq.push(1, 1, 1);
    fq.replaceif(2, 1);
    REQUIRE(fq.contains(1));
    REQUIRE_FALSE(fq.contains(2));

    // Try to replace with higher cost
    fq.replaceif(1, 2);
    REQUIRE(fq.contains(1));
    State<int> state = fq.pop();
    REQUIRE(state.getValue() == 1);
    REQUIRE(state.getFCost() == 2);
    REQUIRE(state.getPathCost() == 1);

    // Try to replace with lower cost
    fq.push(1, 2, 2);
    fq.replaceif(1, 1);
    REQUIRE(fq.contains(1));
    state = fq.pop();
    REQUIRE(state.getValue() == 1);
    REQUIRE(state.getFCost() == 3);
    REQUIRE(state.getPathCost() == 1);

    std::cout<<"success\n";
}
