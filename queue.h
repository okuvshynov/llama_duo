#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class mt_queue {
private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable cond;

public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(std::move(value));
        cond.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this] { return !queue.empty(); });
        T value = std::move(queue.front());
        queue.pop();
        return value;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }
};

template <typename T, typename Compare = std::less<T>>
class mt_priority_queue {
private:
    std::priority_queue<T, std::vector<T>, Compare> pqueue;
    mutable std::mutex mutex;
    std::condition_variable cond;

public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex);
        pqueue.push(std::move(value));
        cond.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this] { return !pqueue.empty(); });
        T value = std::move(pqueue.top());
        pqueue.pop();
        return value;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return pqueue.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return pqueue.size();
    }
};
