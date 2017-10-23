#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>
#include <algorithm>

int main()
{
    std::string vals;
    std::ifstream ifs( "/Users/feng/workspace/cunn/Astigmatism/Isim_00_a.txt" );

    while ( std::getline( ifs, vals, '\n' ) )
    {
        std::vector<float> v;
        std::stringstream iss;
        iss.str( vals );

        std::copy( std::istream_iterator<float>( iss ), std::istream_iterator<float>(), std::back_inserter( v ) );
        std::cout << "Reading " << v.size() << " elements\n";
    }

    return 0;
}

