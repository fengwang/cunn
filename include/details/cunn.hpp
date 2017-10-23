#ifndef DETAILS_HPP_INCLUDED_DSFPOIJSD3928UYASIUOJHOIUHSDFKJHVJHASDFIUH32IUFFFFE
#define DETAILS_HPP_INCLUDED_DSFPOIJSD3928UYASIUOJHOIUHSDFKJHVJHASDFIUH32IUFFFFE

#include "./utility.hpp"
#include "./device_matrix_view/device_matrix_view.hpp"
#include "./cuda/cuda.hpp"
#include "./cuda/cuda_memory_cache.hpp"
#include "./host_matrix/host_matrix.hpp"
#include "./cuda/cuda_memory_manager.hpp"
#include "./activation.hpp"

#include <cumar.hpp>

#if 0
All matrices dimensions in a neurla network, with Topology vector 't[N]' and m as batch size (or instances), True input TO, True output TI

Layer index:                0,                  1,                   2,              ......             n,              ......                  N-1
I: Input Layer:            (0,0)               (t[1],m)             (t[2],m)         ......             (t[n],m)        ......                  (t[N-1],m)
O: Output Layer:           (0,0)               (t[1],m)           (t[2]+1,m)         ......             (t[n]+1,m)      ......                  (t[N-1],m)
D: \Delta:                 (0,0)               (t[1],m)             (t[2],m)         ......             (t[n],m)        ......                  (t[N-1],m)
W: weight:                 (0,0)               (t[1],t[0])        (t[2],t[1])        ......             (t[n],t[n-1])   ......                  (t[N-1],t[N-2])
G: delta weight:           (0,0)               (t[1],t[0])        (t[2],t[1])        ......             (t[n],t[n-1])   ......                  (t[N-1],t[N-2])

Forward:

    I[n] = W[n] O[n-1]  ---- n \in [1, N-1]
    O[n] = f(I[n])      ---- n \in [1, N-1]

Backward:

    D[n] = O[n] - TO .* f'(I[n])        ---- n = N-1
    D[n] = W'[n+1] D[n+1] .* f'(I[n])   ---- n = N-2, N-1, ..., 1
    G[n] = D[n] O'[n-1]                 ---- n = N-2, N-1, ..., 1

#endif

extern "C"
{
    // returns \sum_i^n |x_i|
    float nrm2( float const* x, unsigned long n );
}

namespace cunn
{
    struct neural_network
    {
        std::vector<unsigned long>                                  topology_;
        std::vector<std::string>                                    activation_;

        cuda_memory_manager                                         weight_manager_;
        std::vector<device_matrix_view<float>>                      weight_;

        neural_network() {}

        neural_network( std::vector<unsigned long> const& topology )
        {
            initialize( topology );
        }

        neural_network( std::vector<unsigned long> const& topology, std::vector<std::string> const& activation ) : neural_network( topology )
        {
            activation_ = activation;

            assert( activation_.size() == topology_.size() );
            for ( auto const& a : activation_ )
            {
                assert( std::any_of( activation_functions.begin()+1, activation_functions.end(), [&a]( std::string const& a_ ){ return a == a_; } ) );
            }
        }

        void initialize( std::vector<unsigned long> const& topology )
        {
            assert( topology.size() > 1 && "too small topology size!" );
            topology_ = topology;
            topology_[0] += 1; //adding bias for 1st layer

            // assemble weight matrices
            weight_.emplace_back(); // <-- first weight is null
            for ( unsigned long idx = 1UL; idx != topology_.size(); ++idx )
                weight_.emplace_back( topology_[idx], topology_[idx-1] );

            weight_manager_.manage( std::make_tuple( weight_.begin(), weight_.end() ) );
            //auto start_address = weight_manager_.address_;
            //auto element = weight_manager_.size_;

            cuda_uniform_initialize( weight_manager_.address_, weight_manager_.size_ ); // randomrize all weights to range [-1, 1]
            //cumar::map()()("[](float&x){x=x*2.0f-1.0f;}")( weight_manager_.address_, weight_manager_.address_+weight_manager_.size_ ); //rescale to [0,1]
            cumar::map()()("[](float&x){x=(x-0.5f)*2.4494897427831780982f;}")( weight_manager_.address_, weight_manager_.address_+weight_manager_.size_ ); //rescale to [0,1]

            activation_.resize( topology_.size() );
            std::fill( activation_.begin(), activation_.end(), "logistic" );
        }

        float train( host_matrix<float> const& train_input, host_matrix<float> const& train_output, unsigned long loops, float regulation = 0.01f )
        {
            std::ofstream ofs( "Residual.txt" );

            assert( train_input.size() );
            assert( train_input.row() == topology_[0] - 1 );
            assert( train_output.row() == (*(topology_.rbegin())) );

            //-------------------------------
            //-----Allocate GPU Memory-------
            //-------------------------------
            unsigned long const instance = train_input.col();
            unsigned long const layers = topology_.size();
            unsigned long const max_dim = *(std::max_element( topology_.begin(), topology_.end() ) );
            unsigned long const weight_size = weight_manager_.size_;

            std::vector<device_matrix_view<float>> weight_gradient;
            for ( auto const& w : weight_ )     // <- this accounts weight_gradient_ size
                weight_gradient.emplace_back( w.row(), w.col() );

            std::vector<device_matrix_view<float>> layer_input;
            layer_input.emplace_back(); // <- first of layer_input is null
            for ( auto idx = 1UL; idx != topology_.size(); ++idx )
                layer_input.emplace_back( topology_[idx], instance );

            std::vector<device_matrix_view<float>> layer_output;
            for ( auto const layer_size : topology_ )
                layer_output.emplace_back( layer_size, instance );

            device_matrix_view<float> true_output{ topology_[layers-1], instance };
            device_matrix_view<float> first_delta{ max_dim, instance }; // place holder for \delta_{n+1}
            device_matrix_view<float> second_delta{ max_dim, instance }; // place holder for \delta_n
            device_matrix_view<float> wt_delta{ max_dim, instance };
            device_matrix_view<float> difference{ topology_[layers-1], instance };
            device_matrix_view<float> activation_derivative{ max_dim, instance };
            device_matrix_view<float> momentum_1st{ weight_manager_.size_, 1 };
            device_matrix_view<float> momentum_2nd{ weight_manager_.size_, 1 };

            cuda_memory_manager training_memory_manager;
            training_memory_manager.manage( std::make_tuple( weight_gradient.begin(), weight_gradient.end() ),
                                            std::make_tuple( layer_input.begin(), layer_input.end() ),
                                            std::make_tuple( layer_output.begin(), layer_output.end() ),
                                            std::addressof( true_output ),
                                            std::addressof( first_delta ),
                                            std::addressof( second_delta ),
                                            std::addressof( wt_delta ),
                                            std::addressof( difference ),
                                            std::addressof( activation_derivative),
                                            std::addressof( momentum_1st ),
                                            std::addressof( momentum_2nd )
                                         );

            device_matrix_view<float> weight_gradient_cache{ weight_size, 1, weight_gradient[0].data() };
            device_matrix_view<float> weight_cache{ weight_size, 1, weight_[0].data() };

            //----------------------------------
            //-----Initialization---------------
            //----------------------------------
            cumar::map()()( "[](float&x){ x = 0.0f; }" ) ( momentum_1st.begin(), momentum_1st.end() );
            cumar::map()()( "[](float&x){ x = 0.0f; }" ) ( momentum_2nd.begin(), momentum_2nd.end() );
            //input for the first layer
            host_to_device( train_input.data(), train_input.data()+train_input.size(), layer_output[0].begin() );
            cumar::map()()("[](float&x){x=1.0f;}")(layer_output[0].begin()+train_input.size(), layer_output[0].end());
            //true output
            host_to_device( train_output.data(), train_output.data()+train_output.size(), true_output.begin() );
            float const total_energy = nrm2( true_output.begin(), true_output.size() );;

            auto peer_keeper = make_peer_access( training_memory_manager.id_, weight_manager_.id_ ); //<- for multiple GPUs

            //parameters for adam
            float const epsilon = 1.0e-8f;
            float const alpha = 0.001f;
            float const beta_1 = 0.9f;
            float const beta_2 = 0.999f;
            float beta_1t = 0.9f;
            float beta_2t = 0.999f;
            float loss_rate = 0.0f; //performance of the nn

            for ( auto idx = 0UL; idx != loops; ++idx )
            {
                //--------------------
                //------forward------
                //--------------------
                for ( auto fdx = 1UL; fdx != topology_.size(); ++fdx )
                {
                    gemm_ab_c( weight_[fdx], layer_output[fdx-1], layer_input[fdx] );
                    cumar::map()()(activation_code.at(activation_[fdx]+std::string{"_activation"}))
                              (layer_input[fdx].begin(), layer_input[fdx].end(), layer_output[fdx].begin());
                }

                //--------------------
                //------backward------
                //--------------------
                difference.resize( topology_[layers-1], instance );
                cumar::map()()("[](float a, float b, float& c){c = a-b;}") // calculate difference
                          (layer_output[layers-1].begin(),layer_output[layers-1].end(),true_output.begin(), difference.begin());

                float const energy_loss = nrm2( difference.begin(), difference.size() );

                std::cout << "Residual at step " << idx << " is " << energy_loss / total_energy << std::endl;
                loss_rate = energy_loss / total_energy;
                ofs << loss_rate << std::endl;

                if ( loss_rate < 0.01f ) break;

                // calculate laster layer input derivative
                activation_derivative.resize( topology_[layers-1], instance );
                cumar::map()()(activation_code.at(activation_[layers-1]+"_derivative"))
                          (layer_input[layers-1].begin(), layer_input[layers-1].end(), activation_derivative.begin());

                // calculate \delta[n+1]
                first_delta.resize(topology_[layers-1], instance); // reshape first delta
                cumar::map()()("[](float a, float b, float&c){ c = a*b; }")
                          (difference.begin(), difference.end(), activation_derivative.begin(), first_delta.begin());

                //gradient at the last pos
                gemm_abT_c( first_delta, layer_output[topology_.size()-2], weight_gradient[topology_.size()-1] );

                for ( auto ddx = topology_.size()-2; ddx != 0; --ddx )
                {
                    // calculate f'(x_input) at layer ddx
                    activation_derivative.resize( topology_[ddx], instance );
                    cumar::map()()(activation_code.at(activation_[ddx]+"_derivative"))
                              (layer_input[ddx].begin(), layer_input[ddx].end(), activation_derivative.begin());
                    // cache of W^T[n+1] \delta[n+1]
                    wt_delta.resize( topology_[ddx], instance );
                    gemm_aTb_c( weight_[ddx+1], first_delta, wt_delta );
                    second_delta.resize( topology_[ddx], instance );
                    cumar::map()()("[](float a, float b, float& c){ c = a*b; }")
                              (wt_delta.begin(), wt_delta.end(), activation_derivative.begin(), second_delta.begin());

                    // \delta_w[n] = \delta[n] layer_out^T[n-1]
                    gemm_abT_c( second_delta, layer_output[ddx-1], weight_gradient[ddx] );

                    // second_delta -> first_delta
                    first_delta.swap( second_delta );
                }


                //TODO: calculate conjugate gradient
                {
                }


                //--------------------------
                //------Update Weights------
                //--------------------------
                //
                // m = \beta_1 m + (1-\beta_1) dw       <<-- first momentum
                cumar::map()( "beta_1", beta_1 )( "[](float& m, float dw){ m = beta_1 * m + (1.0f-beta_1)*dw; }" )
                    ( momentum_1st.begin(), momentum_1st.end(), weight_gradient_cache.begin() );
                // v = \beta_2 v + (1-\beta_2) dw dw    <<-- second momentum
                cumar::map()( "beta_2", beta_2 )( "[](float& v, float dw){ v = beta_2 * v + (1.0f-beta_2)*dw*dw; }" )
                    ( momentum_2nd.begin(), momentum_2nd.end(), weight_gradient_cache.begin() );
                // w -= \alpha m / ( (1-\beta_1t) * sqrt(v/(1+b2t)+eps) )
                cumar::map()( "alpha", alpha, "beta_1t", beta_1t, "beta_2t", beta_2t, "epsilon", epsilon )
                    ("[](float&w, float m, float v){ w -= alpha * m / ( (1.0f-beta_1t) * sqrtf( v/(1+beta_2t) + epsilon ) ); }")
                    ( weight_cache.begin(), weight_cache.end(), momentum_1st.begin(), momentum_2nd.begin() );
                // \beta_1t *= \beta_1
                beta_1t *= beta_1;
                // \beta_2t *= \beta_2
                beta_2t *= beta_2;
            }

            ofs.close();

            return loss_rate;
        }

        host_matrix<float> const predict( host_matrix<float> const& predict_input )
        {
            assert( predict_input.size() );
            assert( predict_input.row() == topology_[0] - 1 );

            cuda_memory_manager predict_memory_manager;

            unsigned long const instance = predict_input.col();
            unsigned long const layers = topology_.size();

            // allocate memory cache
            std::vector<device_matrix_view<float>> layer_input;
            layer_input.emplace_back(); // <- first of layer_input is null
            for ( auto idx = 1UL; idx != topology_.size(); ++idx )
                layer_input.emplace_back( topology_[idx], instance );

            // construct layer_output
            std::vector<device_matrix_view<float>> layer_output;
            for ( auto idx = 0UL; idx != topology_.size(); ++idx )
                layer_output.emplace_back( topology_[idx], instance );

            predict_memory_manager.manage(  std::make_tuple(layer_output.begin(), layer_output.end()),
                                            std::make_tuple(layer_input.begin(), layer_input.end())
                                         );

            // input as the output of the first layer
            host_to_device( predict_input.data(), predict_input.data()+predict_input.size(), layer_output[0].begin() );
            cumar::map()()("[](float&x){x=1.0f;}")(layer_output[0].begin()+predict_input.size(), layer_output[0].end());

            auto peer_keeper = make_peer_access( predict_memory_manager.id_, weight_manager_.id_ ); //<- for multiple GPUs

            for ( auto fdx = 1UL; fdx != topology_.size(); ++fdx )
            {
                gemm_ab_c( weight_[fdx], layer_output[fdx-1], layer_input[fdx] );
                cumar::map()()(activation_code.at(activation_[fdx]+std::string{"_activation"}))(layer_input[fdx].begin(), layer_input[fdx].end(), layer_output[fdx].begin());
            }

            host_matrix<float> prediction{ *(topology_.rbegin()), instance };
            device_to_host( (*(layer_output.rbegin())).data(), (*(layer_output.rbegin())).data()+prediction.size(), prediction.data() );

            return prediction;
        }

        double validate( host_matrix<float> const& validate_input, host_matrix<float> const& validate_output )
        {

            assert( validate_input.size() && "Input matrix empty" );
            assert( validate_input.row() == topology_[0] -1 && "Input size not match with network topology" ); //we increase one at the initialization phase
            assert( validate_output.row() == (*(topology_.rbegin())) && "Output size not match with network topology" );

            auto prediction = predict( validate_input );
            prediction -= validate_output;

            return std::sqrt(  std::inner_product( prediction.begin(), prediction.end(), prediction.begin(), 0.0 ) /
                               std::inner_product( validate_output.begin(), validate_output.end(), validate_output.begin(), 0.0 )
                            );
        }

        void save_as( std::string const& file_path ) const
        {
            std::ofstream ofs( file_path.c_str() );
            assert( ofs && "save_as:: failed to open file." );

            std::copy( topology_.begin(), topology_.end(), std::ostream_iterator<unsigned long>( ofs, "\t") );
            ofs << "\n";

            std::copy( activation_.begin(), activation_.end(), std::ostream_iterator<std::string>( ofs, "\t") );
            ofs << "\n";

            unsigned long total_size = 0;
            for ( auto const& w : weight_ ) total_size += w.size();
            std::vector<float> weight;
            weight.resize( total_size );
            device_to_host( weight_[0].data(), weight_[0].data()+total_size, weight.data() );

            ofs.precision( 15 );
            std::copy( weight.begin(), weight.end(), std::ostream_iterator<float>( ofs, "\t") );
            ofs << "\n";

        }

        bool load( std::string const& file_path )
        {
            std::ifstream ifs( file_path.c_str() );
            assert( ifs && "load:: failed to open file." );

            std::stringstream iss;
            std::string cache;

            //load topology
            std::getline( ifs, cache );
            assert( ifs.good() && "load:: failed to load network topology." );
            iss.str(cache);
            std::copy( std::istream_iterator<unsigned long>(iss), std::istream_iterator<unsigned long>(), std::back_inserter(topology_) );

            // assemble weight matrices
            weight_.emplace_back(); // <-- first weight is null
            for ( unsigned long idx = 1UL; idx != topology_.size(); ++idx )
                weight_.emplace_back( topology_[idx], topology_[idx-1] );

            weight_manager_.manage( std::make_tuple( weight_.begin(), weight_.end() ) );

            auto const&split = [](const std::string &text, char sep)
            {
                std::vector<std::string> tokens;
                std::size_t start = 0, end = 0;
                while ((end = text.find(sep, start)) != std::string::npos)
                {
                    tokens.push_back(text.substr(start, end - start));
                    start = end + 1;
                }
                auto const xx = text.substr(start);
                if ( xx != "\t" && xx != " " && xx.size() )
                    tokens.push_back(xx);
                return tokens;
            };

            //load activation
            std::getline( ifs, cache );
            assert( ifs.good() && "load:: failed to load network activation." );
            activation_ = split( cache, '\t');

            /*
            std::for_each( cache.begin(), cache.end(), [](char& c) { if (c == '\t') c = '\n'; } );
            iss.str(cache);
            //NOT work here .....
            std::copy( std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(), std::back_inserter(activation_) );
            */
            assert( activation_.size() == topology_.size() && "Activation size not match with topology." );


            //load weight
            std::getline( ifs, cache );
            assert( ifs.good() && "load:: failed to load network weight." );
            auto const& ws = split( cache, '\t' );
            std::vector<float> weight;
            for (auto const& w : ws )
            {
                weight.push_back( std::stof(w) );
            }

            /*
            iss.str(cache);
            std::cout << "Loaded: " << cache << std::endl;
            //no idea why this fail
            std::vector<float> weight{ std::istream_iterator<float>{iss}, std::istream_iterator<float>{} };
            */

            assert( weight.size() == weight_manager_.size_ && "load:: weight size not match." );

            host_to_device( weight.data(), weight.data()+weight_manager_.size_, weight_manager_.address_ );

            return true;
        }

    };//struct

    std::ostream& operator << ( std::ostream& os, neural_network const& nn )
    {
        os << "Topology:\n";
        std::copy( nn.topology_.begin(), nn.topology_.end(), std::ostream_iterator<unsigned long>(os, "\t") );
        os << "\n";

        os << "Activation:\n";
        std::copy( nn.activation_.begin(), nn.activation_.end(), std::ostream_iterator<std::string>(os, "\t") );
        os << "\n";

        os << "Weight:\n";
        for ( auto const& w : nn.weight_ )
            os << w << "\n";

        os << std::endl;

        return os;
    }
}

 #endif//DETAILS_HPP_INCLUDED_DSFPOIJSD3928UYASIUOJHOIUHSDFKJHVJHASDFIUH32IUFFFFE
