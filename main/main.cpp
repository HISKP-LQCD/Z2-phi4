#define CONTROL

#include <array>
#include <cstring> 
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>


#include "IO_params.hpp"
#include "mutils.hpp"
#include "lattice.hpp"
#include "geometry.hpp"
#include "updates.hpp"
#include <random>

#include "random.hpp"
#include "utils.hpp" 
#include "write_viewer.hpp"
#include "measurements.hpp"
#include "DFT.hpp"

//#include <highfive/H5File.hpp>

#include <Kokkos_Core.hpp>





////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

    endian = endianness();
#ifdef DEBUG
    printf("DEBUG mode ON\n");
#endif
#ifdef FFTW
    cout << "FFTW: ON " << endl;
#endif
    printf("git commit %s\n", GIT_COMMIT_HASH);

    printf("endianness=%d  (0 unknown , 1 little , 2 big)\n", endian);
    if (endian == UNKNOWN_ENDIAN) { printf("UNKNOWN_ENDIAN abort\n"); exit(0); }

    cluster::IO_params params(argc, argv);
    cout << "time " << params.data.L[0] << endl;
    cout << "volume " << params.data.V << endl;
    cout << "seed " << params.data.seed << endl;
    cout << "level " << params.data.level << endl;
    cout << "output " << params.data.outpath << endl;
    cout << "start mes " << params.data.start_measure << endl;

    cout << "metropolis_local_hits " << params.data.metropolis_local_hits << endl;
    cout << "metropolis_global_hits " << params.data.metropolis_global_hits << endl;
    cout << "metropolis_delta  " << params.data.metropolis_delta << endl;
    size_t V = params.data.V;

    printf("save_config = %s\n", params.data.save_config.c_str());
    printf("save_config_FT = %s\n", params.data.save_config_FT.c_str());
    printf("compute_contractions = %s\n", params.data.compute_contractions.c_str());




    // init_rng( params);

    // starting kokkos
    Kokkos::initialize(argc, argv); {
        Kokkos::Timer timer;

        cout << "Kokkos started:" << endl;
        cout << "   execution space:" << typeid(Kokkos::DefaultExecutionSpace).name() << endl;
        cout << "   host  execution    space:" << &Kokkos::HostSpace::name << endl;

        int layout_value = check_layout();
        // Create a random number generator pool (64-bit states or 1024-bit state)
        // Both take an 64 bit unsigned integer seed to initialize a Random_XorShift generator 
        // which is used to fill the generators of the pool.
        //RandPoolType rand_pool(params.data.seed);
        RandPoolType rand_pool;
        rand_pool.init(params.data.seed, V / 2);
        //Kokkos::Random_XorShift1024_Pool<> rand_pool1024(5374857); 
        cout << "random pool initialised" << endl;


        // we need a random generator on the host for the cluster
        // seed the PRNG (MT19937) for each  lattice size, with seed , CPU only
        std::mt19937_64 host_rand(params.data.seed);


        ViewLatt    sectors(params.data.V / 2);
        {
            Kokkos::View<size_t**>    hop("hop", V, 2 * dim_spacetime);
            Kokkos::View<size_t**>    ipt("ipt", V, dim_spacetime);
            hopping(params.data.L, hop, sectors, ipt);
        }
        cout << "hopping initialised" << endl;

        Viewphi  phi("phi", 2, V);
        Viewphip  s_phi("s_phi", 2, V);
#ifdef DEBUG
        test_FT(params);
#ifdef FFTW
        test_FT_vs_FFTW(params);
#endif
#endif


        // Initialize phi on the device
        Kokkos::parallel_for("init_phi", V / 2, KOKKOS_LAMBDA(size_t x) {
            // get a random generatro from the pool
            gen_type rgen = rand_pool.get_state(x);

            phi(0, x) = (rgen.drand() * 2. - 1.);
            phi(1, x) = (rgen.drand() * 2. - 1.);

            phi(0, x + V / 2) = (rgen.drand() * 2. - 1.);
            phi(1, x + V / 2) = (rgen.drand() * 2. - 1.);
            // Give the state back, which will allow another thread to aquire it
            rand_pool.free_state(rgen);
        });
        if (V % 2 == 1) {
            Kokkos::parallel_for("init_phi", 1, KOKKOS_LAMBDA(size_t x) {
                // get a random generatro from the pool
                gen_type rgen = rand_pool.get_state(x);
                phi(0, V - 1) = (rgen.drand() * 2. - 1.);
                phi(1, V - 1) = (rgen.drand() * 2. - 1.);
                // Give the state back, which will allow another thread to aquire it
                rand_pool.free_state(rgen);
            });
        }

        /*
        Viewphi phip("phip",2,params.data.L[0]*Vp);
        Viewphi::HostMirror h_phip= Kokkos::create_mirror_view( phip );

        complexphi cphip("complex_phip",2,params.data.L[0]*Vp/2);
        complexphi::HostMirror h_cphip= Kokkos::create_mirror_view( cphip );

        complexphi s_cphip("complex_phip",2,params.data.L[0]*Vp/2);
        complexphi::HostMirror h_s_cphip= Kokkos::create_mirror_view( s_cphip );

        complexphi cphi2p("complex_phi2p",2,params.data.L[0]*Vp/2);
        */
        // the following ordering is important
        Npfileds = 1;
        if (params.data.smearing == "yes")
            Npfileds = 2;
        if (params.data.FT_phin == "yes")
            Npfileds = 4;
        if (params.data.smearing3FT == "yes")
            Npfileds = 5;

        std::cout << "Npfileds = " << Npfileds << std::endl;

        manyphi mphip("manyphi", Npfileds, 2, params.data.L[0] * Vp * 2); // ( " phi, smeared, phi2, phi3" , comp, "t+p*T") 
        manyphi::HostMirror h_mphip;
        if (params.data.save_config_FT_bundle == "yes" || params.data.save_config_FT == "yes" || params.data.checks == "yes")  h_mphip = Kokkos::create_mirror_view(mphip);

        std::string suffix = "_T" + std::to_string(params.data.L[0]) +
            "_L" + std::to_string(params.data.L[1]) +
            "_msq0" + std::to_string(params.data.msq0) + "_msq1" + std::to_string(params.data.msq1) +
            "_l0" + std::to_string(params.data.lambdaC0) + "_l1" + std::to_string(params.data.lambdaC1) +
            "_mu" + std::to_string(params.data.muC) + "_g" + std::to_string(params.data.gC) +
            "_rep" + std::to_string(params.data.replica);

        std::string mes_file = params.data.outpath + "/mes" + suffix;

        std::string G2t_file = params.data.outpath + "/G2t" + suffix;
        FILE* f_checks = NULL;
        if (params.data.checks == "yes") {
            std::string checks_file = params.data.outpath + "/checks" + suffix;
            f_checks = fopen(checks_file.c_str(), "w+");
            if (f_checks == NULL) {
                printf("Error opening file %s  \n", checks_file.c_str());
                exit(1);
            }
            write_header_measuraments(f_checks, params, 15);
        }

        cout << "Writing magnetization to: " << mes_file << endl;
        cout << "Writing G2t       to: " << G2t_file << endl;
        FILE* f_mes = fopen(mes_file.c_str(), "w+");
        FILE* f_G2t = fopen(G2t_file.c_str(), "w+");
        if (f_mes == NULL || f_G2t == NULL) {
            printf("Error opening file %s or %s \n", mes_file.c_str(), G2t_file.c_str());
            exit(1);
        }
        write_header_measuraments(f_G2t, params);

        FILE* f_conf_bundle = NULL;
        if (params.data.save_config_FT_bundle == "yes") {
            std::string conf_file = params.data.outpath +
                "/T" + std::to_string(params.data.L[0]) + "_L" + std::to_string(params.data.L[1]) +
                "_msq0" + std::to_string(params.data.msq0) + "_msq1" + std::to_string(params.data.msq1) +
                "_l0" + std::to_string(params.data.lambdaC0) + "_l1" + std::to_string(params.data.lambdaC1) +
                "_mu" + std::to_string(params.data.muC) + "_g" + std::to_string(params.data.gC) +
                "_rep" + std::to_string(params.data.replica) +
                "_conf_FT_bundle";
            f_conf_bundle = fopen(conf_file.c_str(), "w+");
            if (f_conf_bundle == NULL) {
                printf("Error opening file %s!\n", conf_file.c_str());
                Kokkos::abort("opening file");
            }
            cout << "Writing all the FT config  to: " << conf_file.c_str() << endl;
            write_header(f_conf_bundle, params);
            int Nmeas = params.data.total_measure / params.data.measure_every_X_updates;
            fwrite(&Nmeas, sizeof(int), 1, f_conf_bundle);
        }

        double time_update = 0, time_mes = 0, time_writing = 0, time_FT = 0;
        int nFT = 0;
        double ave_acc = 0;
        // The update ----------------------------------------------------------------
        for (int ii = 0; ii < params.data.start_measure + params.data.total_measure; ii++) {
            // Timer 
            Kokkos::Timer timer1;
            double time;
            //// // cluster update
            //// double cluster_size = 0.0;
            //// for(size_t nb = 0; nb < params.data.cluster_hits; nb++)
            ////     cluster_size += cluster_update(  phi ,  params, rand_pool, host_rand ,hop  );
            //// cluster_size /= params.data.cluster_hits;
            //// cluster_size /= (double) V;

            time = timer1.seconds();
            //printf("time cluster (%g  s)   size=%g\n",time,cluster_size);
            // metropolis update
            double acc = 0.0;
            for (int global_metro_hits = 0; global_metro_hits < params.data.metropolis_global_hits; global_metro_hits++) {
                acc += metropolis_update(phi, params, rand_pool, sectors);
                modulo_2pi(phi, params.data.V);
            }
            acc /= (params.data.metropolis_global_hits);
            ave_acc += acc / ((double)V);
            // cout << "Metropolis.acc=" << acc/V << endl ;

             // Calculate time of Metropolis update
            time = timer1.seconds();
            time_update += time;

            // bool condition in the infile 
            bool measure = (ii >= params.data.start_measure && (ii - params.data.start_measure) % params.data.measure_every_X_updates == 0);
            bool contractions = (measure && params.data.compute_contractions == "yes");
            bool write_FT = (measure && params.data.save_config_FT == "yes");
            bool write_FT_bundle = (measure && params.data.save_config_FT_bundle == "yes");
            bool write = (measure && params.data.save_config == "yes");

            if (contractions || write_FT || write_FT_bundle) {
                Kokkos::Timer timer_FT;

#ifndef cuFFT   
                compute_FT_complex(mphip, 0, phi, params, 1);
                if (params.data.smearing == "yes") {
                    smearing_field(s_phi, phi, params);
                    compute_FT_complex_smearing(mphip, 1, s_phi, params, 1);
                }
                if (params.data.FT_phin == "yes") {
                    compute_FT_complex(mphip, 2, phi, params, 2);
                    compute_FT_complex(mphip, 3, phi, params, 3);
                }
                if (params.data.smearing3FT == "yes") {
                    compute_smearing3FT(mphip, 4, phi, params);
                }
#endif
#ifdef cuFFT   
                //compute_cuFFT(phi, params ,   ii, h_phip);
                // fixme
                Kokkos::abort("cuFFT not supported");
#endif
                if (params.data.checks == "yes")  Kokkos::deep_copy(h_mphip, mphip);

                Kokkos::fence();   // ----------------------fence-------------------------------// 
                time = timer_FT.seconds();
                time_FT += time;
                nFT++;
            }


            //Measure every 
            if (contractions) {
                Kokkos::Timer timer_2;

                double* m = compute_magnetisations(phi, params);
                fprintf(f_mes, "%.15g   %.15g \n", m[0], m[1]);
                free(m);

                parallel_measurement_complex(mphip, h_mphip, params, f_G2t, f_checks, ii);
                // check_spin(phi, params);
                time = timer_2.seconds();
                time_mes += time;
            }

            // write conf FT
            if (write_FT) {
                Kokkos::Timer timer3;
                std::string conf_file = params.data.outpath +
                    "/T" + std::to_string(params.data.L[0]) + "_L" + std::to_string(params.data.L[1]) +
                    "_msq0" + std::to_string(params.data.msq0) + "_msq1" + std::to_string(params.data.msq1) +
                    "_l0" + std::to_string(params.data.lambdaC0) + "_l1" + std::to_string(params.data.lambdaC1) +
                    "_mu" + std::to_string(params.data.muC) + "_g" + std::to_string(params.data.gC) +
                    "_rep" + std::to_string(params.data.replica) +
                    "_conf_FT" + std::to_string(ii);
                FILE* f_conf = fopen(conf_file.c_str(), "w+");
                if (f_conf == NULL) { printf("Error opening file %s!\n", conf_file.c_str());     Kokkos::abort("opening file"); }

                Kokkos::deep_copy(h_mphip, mphip);
                write_conf_FT_complex(f_conf, layout_value, params, ii, h_mphip);

                time = timer3.seconds();
                fclose(f_conf);
                time_writing += time;
            }
            if (write_FT_bundle) {
                Kokkos::Timer timer3;

                Kokkos::deep_copy(h_mphip, mphip);
                write_single_conf_FT_complex(f_conf_bundle, layout_value, params, ii, h_mphip);

                time = timer3.seconds();
                time_writing += time;
            }

            // write the configuration to disk
            if (write) {
                Kokkos::Timer timer3;
                std::string conf_file = params.data.outpath +
                    "/T" + std::to_string(params.data.L[0]) + "_L" + std::to_string(params.data.L[1]) +
                    "_msq0" + std::to_string(params.data.msq0) + "_msq1" + std::to_string(params.data.msq1) +
                    "_l0" + std::to_string(params.data.lambdaC0) + "_l1" + std::to_string(params.data.lambdaC1) +
                    "_mu" + std::to_string(params.data.muC) + "_g" + std::to_string(params.data.gC) +
                    "_rep" + std::to_string(params.data.replica) +
                    "_conf" + std::to_string(ii);
                cout << "Writing configuration to: " << conf_file << endl;
                FILE* f_conf = fopen(conf_file.c_str(), "w+");
                if (f_conf == NULL) {
                    printf("Error opening file %s!\n", conf_file.c_str());
                    exit(1);
                }
                write_viewer(f_conf, layout_value, params, ii, phi);
                time = timer3.seconds();
                fclose(f_conf);
                time_writing += time;
            }
        }



        printf("average acceptance rate= %g\n", ave_acc / (params.data.start_measure + params.data.total_measure));

        printf("  time updating = %f s (%f per single operation)\n", time_update, time_update / (params.data.start_measure + params.data.total_measure));
        printf("  time FT       = %f s (%f per single operation)\n", time_FT, time_FT / (params.data.total_measure / params.data.measure_every_X_updates));
        printf("  time mesuring = %f s (%f per single operation)\n", time_mes, time_mes / (params.data.total_measure / params.data.measure_every_X_updates));
        printf("  time writing  = %f s (%f per single operation)\n", time_writing, time_writing / (params.data.total_measure / params.data.measure_every_X_updates));

        printf("sum time = %f s\n", time_writing + time_mes + time_update + time_FT);


        fclose(f_G2t);
        fclose(f_mes);
        if (params.data.save_config_FT_bundle == "yes") fclose(f_conf_bundle);
        if (params.data.checks == "yes")  fclose(f_checks);
        printf("total kokkos time = %f s\n", timer.seconds());
    }


    Kokkos::finalize();

    return 0;
}
