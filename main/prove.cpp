#define CONTROL

#include <array>
#include <cstring> 
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>

//#add_executable(prove prove.cpp)
//#target_link_libraries(prove  kokkos)


#include <Kokkos_Core.hpp>

#define N 5
KOKKOS_FUNCTION 
void f0(){
    printf("0\n");
}
KOKKOS_FUNCTION 
void f1(){
    printf("1\n");
}
KOKKOS_FUNCTION 
void f2(){
    printf("2\n");
}
KOKKOS_FUNCTION 
void f3(){
    printf("3\n");
}
KOKKOS_FUNCTION 
void f4(){
    printf("4\n");
}


KOKKOS_FUNCTION 
template<int i>
void f(){
    printf("%d\n",i);
}

template void f<1>(){printf("1\n");}

int main(int argc, char** argv){

Kokkos::initialize( argc, argv );{
    
    std::cout << "Kokkos started:"<< std::endl; 
    std::cout << "   execution space:"<< typeid(Kokkos::DefaultExecutionSpace).name() << std::endl; 
    std::cout << "   host  execution    space:"<<  &Kokkos::HostSpace::name << std::endl; 
    Kokkos::Timer timer;
    Kokkos::parallel_for( "measurement_t_loop",N, KOKKOS_LAMBDA( int i) {

        printf("%d\n",i);

    });
    Kokkos::fence();
    printf("kokkos time = %f s\n", timer.seconds());
    printf("\n");

    Kokkos::Timer timer1;
    Kokkos::parallel_for( "measurement_t_loop",N, KOKKOS_LAMBDA( int i) {
        void (*pf[N])();
        pf[0]=f0;
        pf[1]=f1;
        pf[2]=f2;
        pf[3]=f3;
        pf[4]=f4;
        pf[i]();

    });
    Kokkos::fence();
    printf("kokkos time1 = %f s\n", timer1.seconds());


    Kokkos::Timer timer2;
    Kokkos::parallel_for( "measurement_t_loop",N, KOKKOS_LAMBDA( int i) {
        f<i>();

    });
    Kokkos::fence();
    printf("kokkos time1 = %f s\n", timer2.seconds());

    
}
Kokkos::finalize();
}