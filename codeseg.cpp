#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <fstream>
#include <thread>
#include <mutex>
#include <sstream>
#include <exception>
#include <chrono>
#include<cstdlib>

//First Non-Repeating Character
char firstnonrepeatingchar(const string & s){
    std::array<int, 256> count{};
    for (char c : s) { cout[static_cast<uint8_t>(c)]++;}
    for (char c : s){
       if count[static_cast<uint8_t>(c)] == 1
            return c;
    return '\0';
}


struct Node{
    int value;
    Node* next;

};

Node* reverse_list(Node* haed){
    Node* prev = nullptr;
    while(head){
       Node* next = head->next;
       head->next=prev;
       prev=head;
       head = next;
    }
    return prev;
}


bool hasCycle(Node* head) {
    Node *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;          // 1 step
        fast = fast->next->next;    // 2 steps
        if (slow == fast) return true;  // Collision = cycle
    }
    return false;  // Fast reached null = no cycle
}


void norepeating(const string & s){


}


void byteManipulation(){

    uint8_t byte = 0x7;
    printf("0x%02X\n",byte);

}

void printBytes(uint32_t data){

    uint8_t* ptr=reinterpret_cast<uint8_t*>(&data);

    for(int i=0; i<4; i++){
        std::printf("%d 0x%02X\n",i, ptr[i]);
    }
}

bool isLittleIndian()
{
    uint32_t a =1;
    return *reinterpret_cast<uint8_t*>(&a) == 1;
}



// RAII class for file handling
class FileHandler {
    std::ofstream file;
public:
    FileHandler(const std::string& filename) {
        file.open(filename);
        if (!file) throw std::ios_base::failure("Failed to open file");
    }

    void write(const std::string& data) {
        file << data << std::endl;
    }

    ~FileHandler() {
        if (file.is_open()) {
            file.close();
        }
    }
};

// Base class
class Employee {
protected:
    std::string name;
    int id;

public:
    Employee(std::string name, int id) : name(std::move(name)), id(id) {
        std::cout << "Parameter: " << name << std::endl; // name(std::move(name)) causes a move (faster for rvalues, risky to use the original after)
        std::cout << "Member: " << this->name << std::endl;
    }
    virtual void display() const {
        std::cout << "Employee: " << name << ", ID: " << id << "\n";
    }

    virtual ~Employee() = default; // Polymorphic base class needs virtual destructor
};

// Derived class
class Manager : public Employee {
    int teamSize;

public:
    Manager(std::string name, int id, int teamSize)
        : Employee(std::move(name), id), teamSize(teamSize) {}

    void display() const override {
        std::cout << "Manager: " << name << ", ID: " << id << ", Team Size: " << teamSize << "\n";
    }
};

// Template function
template <typename T>
T add(T a, T b) {
    return a + b;
}

// Move semantics demo class
class ResourceHolder {
    std::unique_ptr<int[]> data;
    size_t size;

public:
    ResourceHolder(size_t size) : size(size), data(std::make_unique<int[]>(size)) {
        for (size_t i = 0; i < size; ++i) data[i] = i;
    }

    // Move constructor
    ResourceHolder(ResourceHolder&& other) noexcept
        : data(std::move(other.data)), size(other.size) {
        other.size = 0;
    }

    // Deleted copy constructor
    ResourceHolder(const ResourceHolder&) = delete;

    void show() const {
        for (size_t i = 0; i < size; ++i) std::cout << data[i] << " ";
        std::cout << "\n";
    }
};

// Mutex for multithreading safety
std::mutex cout_mutex;

void threadTask(int id) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "Thread ID: " << id << " is working\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

int main() {
    try {


        std::string s1 = "hello";
        std::string s2 = std::move(s1);  // s1 is an lvalue, std::move turns it into rvalue
        //std::move() casts an lvalue to an rvalue reference

        int a = 10;
        int& lref = a;       // lvalue reference
        int&& rref = 5 + 5;   // rvalue reference
        int* ptr = &a;        // pointer

        // 1. OOP & Polymorphism
        std::vector<std::unique_ptr<Employee>> staff;
        staff.emplace_back(std::make_unique<Employee>("Alice", 101));
        staff.emplace_back(std::make_unique<Manager>("Bob", 102, 5));

        for (const auto& emp : staff) {
            emp->display();
        }

        // 2. STL & Lambda
        std::vector<int> nums = {1, 5, 3, 4, 2};
        std::sort(nums.begin(), nums.end(), [](int a, int b) { return a > b; });
        std::cout << "Sorted Descending: ";
        for (int n : nums) std::cout << n << " ";
        std::cout << "\n";

        // 3. File I/O with RAII
        FileHandler file("output.txt");
        file.write("Writing to file using RAII");

        // 4. Template usage
        std::cout << "Add(3, 4) = " << add(3, 4) << "\n";
        std::cout << "Add(3.5, 2.5) = " << add(3.5, 2.5) << "\n";

        // 5. Move semantics
        ResourceHolder rh1(5);
        ResourceHolder rh2 = std::move(rh1);
        rh2.show();

        // 6. Multithreading
        std::vector<std::thread> threads;
        for (int i = 0; i < 3; ++i)
            threads.emplace_back(threadTask, i);

        for (auto& t : threads)
            t.join();

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << "\n";
    }

    return 0;
}
