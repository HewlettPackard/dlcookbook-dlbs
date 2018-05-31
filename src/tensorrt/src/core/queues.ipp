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

#include <algorithm>

template <typename T>
void abstract_queue<T>::close() { 
    std::unique_lock<std::mutex> lock(m_);
    closed_ = true;
    push_evnt_.notify_all();
    pop_evnt_.notify_all();
}


template <typename T>
void infinite_queue<T>::push(const T& item) throw (queue_closed) {
    std::lock_guard<std::mutex> lock(this->m_);
    if (this->closed_) 
        throw queue_closed();
    item_ = item;
}

template <typename T>
T infinite_queue<T>::pop() throw (queue_closed) {
    std::lock_guard<std::mutex> lock(this->m_);
    if (this->closed_)
            throw queue_closed();
    return item_;
}

template <typename T>
void infinite_queue<T>::empty_queue(std::vector<T>& queue_content) {
    std::lock_guard<std::mutex> lock(this->m_);
    if (!this->closed_ || emptied_)
        return;
    queue_content.clear();
    queue_content.push_back(item_);
    emptied_ = true;
}


template <typename T>
void thread_safe_queue<T>::push(const T& item) throw (queue_closed) {
    std::unique_lock<std::mutex> lock(this->m_);
    if (this->closed_)
        throw queue_closed();
    this->pop_evnt_.wait(
        lock, 
        [this] { return (this->closed_ || max_size_ <=0 || queue_.size() < max_size_); }
    );
    if (this->closed_)
        throw queue_closed();
    queue_.push(item);
    this->push_evnt_.notify_one();
}

template <typename T>
T thread_safe_queue<T>::pop() throw (queue_closed) {
    std::unique_lock<std::mutex> lock(this->m_);
    if (this->closed_)
        throw queue_closed();
    this->push_evnt_.wait(
        lock, 
        [this] { return this->closed_ || !queue_.empty(); }
    );
    if (this->closed_)
        throw queue_closed();
    T item = queue_.front();
    queue_.pop();
    this->pop_evnt_.notify_one();
    return item;
}

template <typename T>
void thread_safe_queue<T>::empty_queue(std::vector<T>& queue_content) {
    std::unique_lock<std::mutex> lock(this->m_);
    if (!this->closed_)
        return;
    queue_content.clear();
    while (!queue_.empty()) {
        queue_content.push_back(queue_.front());
        queue_.pop();
    }
    if (!queue_content.empty())
        std::reverse(queue_content.begin(), queue_content.end());
}
