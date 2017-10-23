#ifndef CUNN_HPP_INCLUDED_DFSPOJAS320UJASJKLHASDL32IJASFDJKASOIJ4398JASDFLKJSALI
#define CUNN_HPP_INCLUDED_DFSPOJAS320UJASJKLHASDL32IJASFDJKASOIJ4398JASDFLKJSALI

#include "./details/cunn.hpp"

namespace cunn
{

    /*
    inline auto make_nn( std::string const& path )
    {
    }
    */

    inline auto make_nn( std::vector<unsigned long> const& dims )
    {
        return neural_network{ dims };
    }

    inline auto make_nn( std::vector<unsigned long> const& dims, std::vector<std::string> const& activations )
    {
        return neural_network{ dims, activations };
    }

    inline void train( neural_network& nn, host_matrix<float> const& input, host_matrix<float> const& output, unsigned long loops = -1, float regulation = 0.01 )
    {
        nn.train( input, output, loops, regulation );
    }

    template< typename Matrix >
    void train( neural_network& nn, Matrix const& input, Matrix const& output, unsigned long loops = -1, float regulation = 0.01 )
    {
        host_matrix<float> input_{ input.row(), input.col() };
        std::copy( input.begin(), input.end(), input_.begin() );

        host_matrix<float> output_{ output.row(), output.col() };
        std::copy( output.begin(), output.end(), output_.begin() );

        return train( nn, input_, output_, loops, regulation );
    }

    inline void train( neural_network& nn, std::string const& input_path, std::string const& output_path, unsigned long loops = -1, float regulation = 0.01 )
    {
        host_matrix<float> input;
        input.load( input_path );
        host_matrix<float> output;
        output.load( output_path );

        train( nn, input, output, loops, regulation );
    }

    inline auto predict( neural_network& nn, host_matrix<float> const& input )
    {
        return nn.predict( input );
    }

    template< typename Matrix >
    inline auto predict( neural_network& nn, Matrix const& input )
    {
        host_matrix<float> input_{ input.row(), input.col() };
        std::copy( input.row(), input.col(), input_ );
        return nn.predict( input_ );
    }

    inline auto predict( neural_network& nn, std::string const& input_path )
    {
        host_matrix<float> input;
        input.load( input_path );
        return predict( nn, input );
    }

    inline auto validate( neural_network& nn, host_matrix<float> const& input, host_matrix<float> const& output )
    {
        return nn.validate( input, output );
    }

    template< typename Matrix >
    auto validate( neural_network& nn, Matrix const& input, Matrix const& output )
    {
        host_matrix<float> input_{ input.row(), input.col() };
        std::copy( input.begin(), input.end(), input_.begin() );

        host_matrix<float> output_{ output.row(), output.col() };
        std::copy( output.begin(), output.end(), output_.begin() );

        return validate( nn, input_, output_ );
    }

    inline auto validate( neural_network& nn, std::string const& input_path, std::string const& output_path )
    {
        host_matrix<float> input;
        input.load( input_path );
        host_matrix<float> output;
        output.load( output_path );
        return validate( nn, input, output );
    }

    inline auto load_nn( std::string const& path )
    {
        neural_network nn;
        nn.load( path );
        return nn;
    }

    inline void save_nn( neural_network const& nn, std::string const& path )
    {
        nn.save_as( path );
    }

}//namespace


#endif//CUNN_HPP_INCLUDED_DFSPOJAS320UJASJKLHASDL32IJASFDJKASOIJ4398JASDFLKJSALI
