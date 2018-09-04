/*
 (c) Copyright [2017] Hewlett Packard Enterprise Development LP
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
// https://juanchopanzacpp.wordpress.com/2013/02/26/concurrent-queue-c11/
// https://codereview.stackexchange.com/questions/149676/writing-a-thread-safe-queue-in-c
#ifndef DLBS_TENSORRT_BACKEND_CORE_QUEUES
#define DLBS_TENSORRT_BACKEND_CORE_QUEUES

#include <exception>
#include <mutex>
#include <queue>
#include <condition_variable>

#define NON_COPYABLE_NOR_MOVABLE(T) \
    T(T const &) = delete; \
    T& operator=(T const &t) = delete; \
    T(T &&) = delete;

/**
 * @brief The exception that is thrown on attempt to push/pop data to/from an empty queue.
 */
class queue_closed : public std::exception {
    const std::string msg = "Queue was closed while performing requested operation.";
public:
    /**
     * @brief Constructor.
     */
    queue_closed() {}
    /**
     * @brief Return exception description.
     */
    const char* what() const noexcept override { return msg.c_str(); }
};

/**
 * @brief An abstract class for all queue implementations.
 */
template <typename T>
class abstract_queue {
protected:
    std::mutex m_;                         //!< Controls exclusive access to queue.
    std::condition_variable push_evnt_;    //!< Signals item was added to a queue.
    std::condition_variable pop_evnt_;     //!< Signals item was removed from a queue.
    bool closed_;                          //!< If true, any operation throws @link queue_closed @endlink exception.
public:
    /**
     * @brief Constructor
     */
    abstract_queue() : closed_(false) {}
    
    NON_COPYABLE_NOR_MOVABLE(abstract_queue)

    void close();
    /**
     * @brief Returns and removes top element from a queue.
     * @return the element from the queue
     * 
     * This call will block if a queue is empty.
     */
    virtual T pop() throw (queue_closed) = 0;
    /**
     * @brief Adds one item to a queue.
     * @param item is the element to push inti the queue,
     * 
     * This call will block if a queue has limited capacity.
     */
    virtual void push(const T& item) throw (queue_closed) = 0;
    /**
     * @brief Returns content of a queue once it's closed.
     */
    virtual void empty_queue(std::vector<T>& queue_content) = 0;
};

/**
 * @brief An infinite queue that always returns one element.
 * 
 * This queue has 'infinite' capacity. The push call over writes
 * element stored in a queue and pop returns that element without
 * 'removing' it from a queue. 
 * Can be used to simulate an input data without actually attaching
 * a real data provider.
 */
template <typename T>
class infinite_queue : public abstract_queue<T> {
private:
    T item_;
    bool emptied_ = false;
public:
    /**
     * @brief Constructor.
     * @param item is the item that will be returned by this queue.
     */
    explicit infinite_queue(const T& item) : abstract_queue<T>() { item_ = item; }
    void push(const T& item) throw (queue_closed) override;
    T pop() throw (queue_closed) override;
    void empty_queue(std::vector<T>& queue_content) override;
};

/**
 * @brief A basic implementation of a thread safe queue.
 */
template <typename T>
class thread_safe_queue : public abstract_queue<T> {
private:
    std::queue<T> queue_;  //!< Actual queue.
    size_t max_size_;      //!< If not 0, defines maximal queue capacity.
public:
    explicit thread_safe_queue(const size_t max_size=0) : abstract_queue<T>(), max_size_(max_size) {}

    void push(const T& item) throw (queue_closed) override;
    T pop() throw (queue_closed) override;
    void empty_queue(std::vector<T>& queue_content) override;
};

#include "core/queues.ipp"

#endif