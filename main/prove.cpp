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

// #define CONCAT(a, b) a ## b
// #define macro(a) CONCAT(func,a)

#include <Kokkos_Core.hpp>

#define N 40
#define T 20

KOKKOS_FUNCTION
void f0(double& r) {
    for (int i = 0; i < T;i++) {
        r += 1;
    }
}
KOKKOS_FUNCTION
void f1(double& r) {
    for (int i = 0; i < T;i++) {
        r += 1;
    }
}
KOKKOS_FUNCTION
void f2(double& r) {
    for (int i = 0; i < T;i++) {
        r += 1;
    }
}
KOKKOS_FUNCTION
void f3(double& r) {
    for (int i = 0; i < T;i++) {
        r += 1;
    }
}
KOKKOS_FUNCTION
void f4(double& r) {
    for (int i = 0; i < T;i++) {
        r += 1;
    }
}
struct BarTag {};
struct RabTag {};
class Foo {
public:
    
    void foobar() {
        printf("foobar = %d\n",0);
    }
    void compute() {
        Kokkos::parallel_for(Kokkos::RangePolicy<BarTag>(0,2), *this);
        Kokkos::parallel_for(Kokkos::RangePolicy<RabTag>(0,3), *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (const BarTag&, const int i) const {
        printf("foobar = bar\n");
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (const RabTag&, const int i) const {
        printf("foobar = Rab\n");
    }

    
};


// template<int i>
// KOKKOS_FUNCTION
// void f() {
//     printf("%d\n", i);
// }

// template void f<1>() { printf("1\n"); }


int main(int argc, char** argv) {

    Kokkos::initialize(argc, argv); {

        std::cout << "Kokkos started:" << std::endl;
        std::cout << "   execution space:" << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
        std::cout << "   host  execution    space:" << &Kokkos::HostSpace::name << std::endl;
        {
            printf("TEST1\n");
            Kokkos::View<double[N][T]> res("res");
            Kokkos::View<double[N][T]>::HostMirror h_res = Kokkos::create_mirror_view(res);
            Kokkos::Timer timer;
            Kokkos::parallel_for("measurement_t_loop", T, KOKKOS_LAMBDA(int t) {
                for (int t1 = 0; t1 < T;t1++) {
                    for (int i = 0; i < N;i++) {
                        res(i, t) += 1;
                    }

                }

            });


            // Kokkos::fence();
            Kokkos::deep_copy(h_res, res);
            printf("kokkos time = %f s\n", timer.seconds());
            for (int i = 0; i < N;i++) {
                if (h_res(i, 0) != T) printf("res[%d]=%g\n", i, h_res(i, 0));
            }
        }
        //////////////

        {
            printf("TEST2\n");
            Kokkos::View<double[N][T]> res("res");
            Kokkos::View<double[N][T]>::HostMirror h_res = Kokkos::create_mirror_view(res);
            Kokkos::Timer timer;
            typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
            typedef Kokkos::TeamPolicy<>::member_type  member_type;

            Kokkos::parallel_for("measurement_t_loop", team_policy(T, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type & teamMember) {
                const int t = teamMember.league_rank();
                for (int i = 0; i < N;i++) {
                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
                        inner += 1;
                        }, res(i, t));
                }

            });
            // Kokkos::fence();
            Kokkos::deep_copy(h_res, res);
            printf("kokkos time = %f s\n", timer.seconds());
            for (int i = 0; i < N;i++) {
                if (h_res(i, 0) != T) printf("res[%d]=%g\n", i, h_res(i, 0));
            }
        }
        //////////////
        {
            printf("TEST3\n");
            Kokkos::View<double[N][T]> res("res");
            Kokkos::View<double[N][T]>::HostMirror h_res = Kokkos::create_mirror_view(res);
            Kokkos::Timer timer;
            typedef Kokkos::TeamPolicy<>               team_policy;//team_policy ( number of teams , team size)
            typedef Kokkos::TeamPolicy<>::member_type  member_type;
            for (int i = 0; i < N;i++) {
                Kokkos::parallel_for("measurement_t_loop", team_policy(T, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type & teamMember) {
                    const int t = teamMember.league_rank();
                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, T), [&](const int t1, double& inner) {
                        inner += 1;
                        }, res(i, t));

                });
            }
            // Kokkos::fence();
            Kokkos::deep_copy(h_res, res);
            printf("kokkos time = %f s\n", timer.seconds());
            for (int i = 0; i < N;i++) {
                if (h_res(i, 0) != T) printf("res[%d]=%g\n", i, h_res(i, 0));
            }
        }

        {
            printf("TEST4\n");
            Kokkos::View<double[N][T]> res("res");
            Kokkos::View<double[N][T]>::HostMirror h_res = Kokkos::create_mirror_view(res);
            Kokkos::Timer timer;
            void (*pf[N])(double&);


            using MDPolicyType_2D = Kokkos::MDRangePolicy<Kokkos::Rank<2> >;
            MDPolicyType_2D mdpolicy_2d({ {0, 0} }, { {N , T} });
            Kokkos::parallel_for("measurement_t_loop", mdpolicy_2d, KOKKOS_LAMBDA(int i, int t) {
                // pf[i](res(i, t));
                switch(i){
                    case 0: f0(res(i, t)); break;
                    case 1: f1(res(i, t)); break;
                    case 2: f2(res(i, t)); break;
                    case 3: f3(res(i, t)); break;
                    case 4: f4(res(i, t)); break;
                    case 5: f0(res(i, t)); break;
                    case 6: f0(res(i, t)); break;
                    case 7: f0(res(i, t)); break;
                    case 8: f0(res(i, t)); break;
                    case 9: f0(res(i, t)); break;
                    case 10: f0(res(i, t)); break;
                    case 11: f0(res(i, t)); break;
                    case 12: f0(res(i, t)); break;
                    case 13: f0(res(i, t)); break;
                    case 14: f0(res(i, t)); break;
                    case 15: f0(res(i, t)); break;
                    case 16: f0(res(i, t)); break;
                    case 17: f0(res(i, t)); break;
                    case 18: f0(res(i, t)); break;
                    case 19: f0(res(i, t)); break;
                    case 20: f0(res(i, t)); break;
                    case 21: f0(res(i, t)); break;
                    case 22: f0(res(i, t)); break;
                    case 23: f0(res(i, t)); break;
                    case 24: f0(res(i, t)); break;
                    case 25: f0(res(i, t)); break;
                    case 26: f0(res(i, t)); break;
                    case 27: f0(res(i, t)); break;
                    case 28: f0(res(i, t)); break;
                    case 29: f0(res(i, t)); break;
                    case 30: f0(res(i, t)); break;
                    case 31: f0(res(i, t)); break;
                    case 32: f0(res(i, t)); break;
                    case 33: f0(res(i, t)); break;
                    case 34: f0(res(i, t)); break;
                    case 35: f0(res(i, t)); break;
                    case 36: f0(res(i, t)); break;
                    case 37: f0(res(i, t)); break;
                    case 38: f0(res(i, t)); break;
                    case 39: f0(res(i, t)); break;
                }
            });
            // Kokkos::fence();
            Kokkos::deep_copy(h_res, res);
            printf("kokkos time = %f s\n", timer.seconds());
            for (int i = 0; i < N;i++) {
                if (h_res(i, 0) != T) printf("res[%d]=%g\n", i, h_res(i, 0));
            }
        }

        Foo  myclass;
        myclass.compute();
        Kokkos::parallel_for(Kokkos::RangePolicy<BarTag>(0,2), myclass);
        // Kokkos::Timer timer1;

        // Kokkos::View<double[N]> res("res");
        // // Kokkos::View<double[N]>::HostMirror h_res = Kokkos::create_mirror_view(res);

        // void (*pf[N])(double&);
        // pf[0] = f0;
        // pf[1] = f1;
        // pf[2] = f2;
        // pf[3] = f3;
        // pf[4] = f4;

        // using MDPolicyType_2D = Kokkos::MDRangePolicy<Kokkos::Rank<2> >;
        // MDPolicyType_2D mdpolicy_2d({ {0, 0} }, { {N , T} });

        // Kokkos::parallel_for("measurement_t_loop", mdpolicy_2d, KOKKOS_LAMBDA(int i, int t) {
        //     printf("measurement_t_loop   %d %d\n",i,t);
        //     pf[i](res(i));
        // });
        // Kokkos::fence();
        // Kokkos::deep_copy(h_res,res);
        // printf("kokkos time1 = %f s\n", timer1.seconds());
        // for (int i = 0; i < N;i++) {
        //     printf("res[%d]=%g\n", i, h_res(i));
        // }



        // Kokkos::Timer timer2;
        // void* (f)();
        // Kokkos::parallel_for("measurement_t_loop", N, KOKKOS_LAMBDA(int i) {
        //     f<i>();

        // });
        // Kokkos::fence();
        // printf("kokkos time1 = %f s\n", timer2.seconds());


    }
    Kokkos::finalize();
}