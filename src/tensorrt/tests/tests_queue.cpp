#include "core/queues.hpp"
#include <chrono>
#include <thread>
#include <iostream>


void client(abstract_queue<int>* queue, long& counter, bool is_provider, int sleep_milliseconds=100);
void test_queue(abstract_queue<int>*q, const int num_providers, const size_t num_consumers);


int main(int /*argc*/, char** /*argv*/) {
    std::cout << "Running Infinite Queue" << std::endl;
    infinite_queue<int> iq(1);
    test_queue(&iq, 10, 20);
    
    std::cout << "Running Thread Safe Queue" << std::endl;
    thread_safe_queue<int> tsq;
    test_queue(&tsq, 10, 20);
    return 0;
}


void client(abstract_queue<int>* queue, long& counter, bool is_provider, int sleep_milliseconds) {
    try {
        while (true) {
            if (is_provider) { queue->push(1); } 
            else { queue->pop(); }
            counter ++;
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_milliseconds));
        }
    } catch(queue_closed) {
    }
}

void test_queue(abstract_queue<int>*q, const int num_providers, const size_t num_consumers) {
    std::vector<std::thread*> providers(num_providers, nullptr),
                              consumers(num_consumers, nullptr);
    std::vector<long> provider_data(num_providers, 0),
                      consumer_data(num_consumers, 0);

    for (size_t i=0; i<num_consumers; ++i)
        consumers[i] = new std::thread(client, std::ref(q), std::ref(consumer_data[i]), false, 100);
    for (int i=0; i<num_providers; ++i)
        providers[i] = new std::thread(client, std::ref(q), std::ref(provider_data[i]), true, 120);

    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    q->close();
    for (size_t i=0; i<num_consumers; ++i) {
        consumers[i] ->join();
        delete consumers[i];
        std::cout << "Consumer " << i << " updated its counter " << consumer_data[i] << " times." << std::endl;
    }
    for (int i=0; i<num_providers; ++i) {
        providers[i] ->join();
        delete providers[i];
        std::cout << "Provider " << i << " updated its counter " << provider_data[i] << " times." << std::endl;
    }

}
