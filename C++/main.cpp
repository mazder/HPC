#include<iostream>
#include<array>
#include<vector>
#include<algorithm> // sort

struct intArray {
    int *iarray;
    intArray(int size) {
        iarray = new int[size];
    }
    ~intArray() {
        std::cout << "Calling struct destructor!\n";
        delete[] iarray;
    }
};


struct Point2D{
    int x,y;
    void print(){
        std::cout<<x<<","<<y<<std::endl;
    }
};

class Point3D{
    public:
        Point3D(int a, int b, int c):x{a},y{b},z{c}
        {
        }
        virtual ~Point3D(){
            std::cout << "Calling destructor 3D!\n";
        }
        //copy constructor

        // Point3D(const Point3D &p) = delete; // no implementation
        Point3D(const Point3D &rh){
            x=rh.x;
            y=rh.y;
            z=rh.z;
        }
        // Operator Overloading
        Point3D& operator=(const Point3D& rh){
            x=rh.x;
            y=rh.y;
            z=rh.z;
            return *this;
        }
        Point3D& operator+=(const Point3D& rh){
            x=x+rh.x;
            y=y+rh.y;
            z=z+rh.z;
            return *this;
        }

        void print(){
            std::cout<<x<<","<<y<<","<<z<<std::endl;
        }
    private:
        int x,y,z;
};

// pass by reference
void print_array(auto& varray) { // requires -std=c++20
    std::cout << "Printing from our general!\n";
    for (auto value : varray) {
        std::cout << value << ' ';
    }
    std::cout << '\n';

    for(auto it=varray.begin(); it<varray.end(); it+=1){
        std::cout<<*it<<' ';
    }
    std::cout<<std::endl;
}

void print_darray(int* varray, int n){
    for(auto i=0; i<n; i+=1){
        std::cout<<varray[i]<<' ';
    }
    std::cout<<std::endl;
}


/*
// pass by value
void print_array(auto varray) { // requires -std=c++20
    std::cout << "Printing from our general!\n";
    for (auto value : varray) {
        std::cout << value << ' ';
    }
    std::cout << '\n';

    for(auto it=varray.begin(); it<varray.end(); it+=1){
        std::cout<<*it<<' ';
    }
    std::cout<<std::endl;
}
*/

/*
template<>
void print_array(std::array<int, 5> varray) {
    std::cout << "Printing from specialization!\n";
    for (auto value : varray) {
        std::cout << value << ' ';
    }
    std::cout << '\n';
}
*/


void copy_constructor_test(Point3D p){
    p.print();
}

// std array
void stdarray_vector(){
    std::array<int, 5> a = {1, 5, 3, 2, 4};
    std::vector<float> v(5,6.f);

    //dynamic allocation
    int *da = new int[10];
    print_darray(da,10);
    delete[] da;

    std::cout<<a.size()<<" "<<v.size()<<std::endl;

    print_array(a);
    print_array(v);

    std::ranges::sort(a);
    print_array(a);

    intArray A(10);
    A.iarray[5]=10;
    print_darray(A.iarray, 10);


    Point2D p;
    p.x=2;
    p.y=3;
    p.print();

    Point3D p3(1,2,3);
    p3.print();

    copy_constructor_test(p3);

    Point3D p4 = p3;

    p4.print();

    p4 += p3;

    p4.print();

}


int main(int argc, char* argv[]){

    stdarray_vector();

    return 0;
}
