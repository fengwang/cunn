#ifndef UTILITY_HPP_INLCUDED_DSPOIJAS23IASJFLKAJSLKSAJD98IJ4LKJDSFLKJFSADLKJSDFF
#define UTILITY_HPP_INLCUDED_DSPOIJAS23IASJFLKAJSLKSAJD98IJ4LKJDSFLKJFSADLKJSDFF

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <future>
#include <initializer_list>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <set>
#include <ctime>

#include "./misc/for_each.hpp"


#ifdef DEBUGLOG
template< typename ... Ts >
void log( Ts&& ... ts )
{
    //auto current_time = std::time(nullptr);
    //std::cout << std::put_time(std::gmtime(std::addressof(current_time)), "%c %Z") << ":\n";
    ((std::cout << std::setbase(16) << std::showbase) << ... << ts ) << std::endl;
}
#else
template< typename ... Ts >
void log( Ts ...)
{
}
#endif

#endif//UTILITY_HPP_INLCUDED_DSPOIJAS23IASJFLKAJSLKSAJD98IJ4LKJDSFLKJFSADLKJSDFF

