#include <cunn.hpp>
using namespace cunn;

int main()
{
    std::vector<unsigned long> dim{ 2, 20, 20, 1 };
    {
        std::cout << "NN topology is set to\n";
        std::copy( dim.begin(), dim.end(), std::ostream_iterator<unsigned long>( std::cout, " " ) );
        std::cout << "\n";
    }
    auto nn = make_nn( dim );
    train( nn, "xor_in.txt", "xor_out.txt", 1280 );
    log( "Training finished with file xor_in.txt and xor_out.txt." );

    std::cout << "Training result\n";

    for ( auto const& w : nn.weight_ )
        std::cout << w << std::endl;


    double res = validate( nn, "xor_in.txt", "xor_out.txt" );
    std::cout << "Residual is " << res << std::endl;


    nn.save_as("xor.nn");
    auto nm = load_nn("xor.nn");


    //std::cout << nm << std::endl;


    double ges = validate( nm, "xor_in.txt", "xor_out.txt" );
    std::cout << "save/load test: validation residual is " << ges << std::endl;


    return 0;
}
