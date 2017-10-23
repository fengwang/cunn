#ifndef RBAXERJBEYLCERDFQTJFOVMUUESBKJJLCNETALSFLEMTXSHYBEWUIBSGYLSKQXIYNHPJSJXRS
#define RBAXERJBEYLCERDFQTJFOVMUUESBKJJLCNETALSFLEMTXSHYBEWUIBSGYLSKQXIYNHPJSJXRS

#include "./typedef.hpp"

namespace cunn
{
    template<typename Matrix, typename Type, typename Allocator>
    struct crtp_io
    {
        typedef Matrix                                                          zen_type;
        typedef crtp_typedef<Type, Allocator>                                   type_proxy_type;
        typedef typename type_proxy_type::size_type                             size_type;
        typedef typename type_proxy_type::difference_type                       difference_type;
        typedef typename type_proxy_type::row_type                              row_type;
        typedef typename type_proxy_type::const_row_type                        const_row_type;
        typedef typename type_proxy_type::reference                             reference;
        typedef typename type_proxy_type::const_reference                       const_reference;
        typedef typename type_proxy_type::value_type                            value_type;


        /// <summary>
        /// matrix output to stream
        /// </summary>
        /// <param name="lhs">
        /// output stream
        /// </param>
        /// <param name="rhs">
        /// a matrix instance
        /// </param>
        /// @return the output stream reference
        ///
        /// Example:
        /// @code{.cpp}
        /// matrix<double> A{ 3, 4 };
        /// std::fill( A.begin(), A.end(), 2 };
        /// std::cout << A << std::endl;
        /// @endcode
        ///
        /// Will produce:
        /// @code{.cpp}
        /// 2    2    2    2
        /// 2    2    2    2
        /// 2    2    2    2
        /// @endcode
        ///
        friend std::ostream& operator << ( std::ostream& lhs, zen_type const& rhs )
        {
            for ( size_type r = 0UL; r != rhs.row(); ++r )
            {
                std::copy( rhs.row_begin(r), rhs.row_end(r), std::ostream_iterator<value_type>( lhs, "\t" ) );
                lhs << "\n";
            }
            return lhs;
        }

        /// <summary>
        /// matrix input from stream
        /// </summary>
        /// <param name="lhs">
        /// input stream
        /// </param>
        /// <param name="rhs">
        /// a matrix instance
        /// </param>
        /// @return the input stream reference
        ///
        friend std::istream& operator >> ( std::istream& lhs, zen_type& rhs )
        {
            std::vector<std::string> row_element;
            std::string string_line;

            while ( std::getline( lhs, string_line, '\n' ) )
                row_element.push_back( string_line );

            size_type const row = row_element.size();
            size_type const col = std::count_if( row_element[0].begin(), row_element[0].end(), []( char ch ) { return '\t' == ch; } );

            if ( row == 0 || col == 0 )
            {
                lhs.setstate( std::ios::failbit );
                return lhs;
            }

            rhs.resize( row, col );

            for ( size_type r = 0; r != row; ++r )
            {
                std::istringstream the_row( row_element[r] );
                std::copy( std::istream_iterator<value_type>( the_row ), std::istream_iterator<value_type>(), rhs.row_begin(r) );
            }

            return lhs;
        }

        /// <summary>
        /// save matrix as a txt file, binary file, bmp file or a  gmp file
        /// </summary>
        /// <param name="path">
        /// The path where the matrix to be save.  If the suffix is '.txt', then the matrix is saved to a txt file; If the suffix is '.bin', then the matrix is saved to a binary file; If the suffix is '.bmp', then the matrix is saved to a bitmap file; If the suffix is '.pgm', then the matrix is saved to a portable gray map file; if the suffix is not specified, then the matrix is saved to a txt file.
        /// </param>
        /// @Return true is saved successfully, false otherwise
        ///
        /// Example:
        /// @code{.cpp}
        /// matrix<unsigned long> M{ 1024, 1024 };
        /// //...
        /// M.save_as( "./m.txt" ); // txt file
        /// M.save_as( "./m.bin" ); // binary file
        /// M.save_as( "./m.bmp" ); // bmp file
        /// M.save_as( "./m.pgm" ); // pgm file
        /// M.save_as( "./matrix" ); // txt file, with file name "./matrix.txt"
        /// M.save_as( "./m_hotblue.bmp", "hotblue" ); // bmp file, but with colormap "hotblue"
        /// M.save_as( "./m_jet.bmp", "jet" ); // bmp file, but with colormap "jet"
        /// M.save_as( "./m_obscure.bmp", "obscure" ); // bmp file, but with colormap "obscure"
        /// M.save_as( "./m_gray.bmp", "gray" ); // bmp file, but with colormap "gray"
        /// @endcode
        bool save_as( std::string const& path ) const
        {
            if ( path.size() < 4 )
                return save_as_txt( path+std::string{".txt"} );

            std::string const& extension{ path.begin()+static_cast<difference_type>(path.size()-4), path.end() };

            if ( extension == std::string{ ".bin" } )
                return save_as_binary( path );

            if ( extension == std::string{".bmp"} )
                return save_as_bmp( path );

            if ( extension == std::string{".pgm"} )
                return save_as_pgm( path );

            if ( extension != std::string{".txt"} )
                return save_as_txt( path + std::string{".txt"} );

            return save_as_txt( path );
        }

        bool save_as( std::string const& path, std::string const& color_map )
        {
            return save_as_bmp( path, color_map );
        }

        bool save_as_txt( std::string const& file_name ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );
            std::ofstream ofs( file_name );

            if ( !ofs ) { return false; }

            ofs.precision( 16 );
            ofs << zen;
            ofs.close();
            return true;
        }

        bool save_as_binary( std::string const& file_name ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );

            std::ofstream ofs( file_name, std::ios::out | std::ios::binary );
            if ( !ofs ) { return false; }

            // store row and col
            auto const r = zen.row();
            ofs.write( reinterpret_cast<char const*>(std::addressof(r)), sizeof(r) );

            auto const c = zen.col();
            ofs.write( reinterpret_cast<char const*>(std::addressof(c)), sizeof(c) );

            // store data as binary
            ofs.write( reinterpret_cast<char const*>(zen.data()), static_cast<std::streamsize>( sizeof(Type)*zen.size() ) );

            if ( !ofs.good() ) return false;

            ofs.close();

            return true;
        }

        bool save_as_bmp( std::string const& file_name, std::string const& color_map = std::string{"default"} ) const
        {
			auto&& make_array = []( unsigned char a, unsigned char b, unsigned char c )
			{
				std::array<unsigned char, 3> ans;
				ans[0] = a;
				ans[1] = b;
				ans[2] = c;
				return ans;
			};

			typedef std::function<std::array<unsigned char, 3>(double)> color_value_type;
			static const std::map<std::string, color_value_type > color_maps
			{
				std::make_pair
				(
					std::string{"default"},
					color_value_type
					{
						[&]( double x )
						{
							typedef unsigned char type;
							auto&& ch = []( double x_ ) { return static_cast<type>( static_cast<int>( x_ * 766.0 ) ); };

							if ( 3.0 * x < 1.0 )
								return make_array( type{0}, type{0}, type{ch(x)} );

							if ( 3.0 * x < 2.0 )
								return make_array( type{0}, type{ ch( x-1.0/3.0 ) }, type{255} );

							return make_array( type{ ch(x-2.0/3.0) }, type{255}, type{255} );
						}
					}
				),
				std::make_pair
				(
					std::string{"hotblue"},
					color_value_type
					{
						[&]( double x )
						{
							typedef unsigned char type;
							auto&& ch = []( double x_ ) { return static_cast<type>( static_cast<int>( x_ * 766.0 ) ); };

							if ( 3.0 * x < 1.0 )
								return make_array( type{ch(1.0/3.0-x)}, type{255}, type{255} );

							if ( 3.0 * x < 2.0 )
								return make_array( type{0}, type{ ch(2.0/3.0-x) }, type{255} );

							return make_array( type{0}, type{0}, type{ch(1.0-x)} );
						}
					}
				),
				std::make_pair
				(
					std::string{"jet"},
					color_value_type
					{
						[&]( double x )
						{
							typedef unsigned char type;
							auto&& ch = []( double x_ ) { return static_cast<type>( static_cast<int>( x_ * 766.0 ) ); };

							if ( 3.0 * x < 1.0 )
								return make_array( type{0}, type{ch(x)}, type{255} );
								//return make_array( type{0}, type{255}, type{ch(x)} );

							if ( 3.0 * x < 2.0 )
								return make_array( type{ch(x-1.0/3.0)}, type{ 255 }, type{ch(2.0/3.0-x)} );

							return make_array( type{ 255 }, type{ch(1.0-x)}, type{0} );
						}
					}
				),
				std::make_pair
				(
					std::string{"obscure"},
					color_value_type
					{
						[&]( double x )
						{
							typedef unsigned char type;
							auto&& ch = []( double x_ ) { return static_cast<type>( static_cast<int>( x_ * 256.0 ) ); };

							type const b = ch( 1.0 - x );

							if ( 4.0 * x < 1 )
								return make_array( ch(1.0-4.0*x), ch(1.0-4.0*x), b );

							type const r = ch( (x - 0.25) * 4.0 / 3.0 );

							if ( 2.0 * x < 1 )
								return make_array( r, ch( (x-0.25)*4.0 ), b );

							return make_array( r, ch( (1.0-x)*2.0 ), b );
						}
					}
				),
				std::make_pair
				(
					std::string{"gray"},
					color_value_type
					{
						[&]( double x )
						{
							typedef unsigned char type;
							auto&& ch = []( double x_ ) { return static_cast<type>( static_cast<int>( x_ * 256.0 ) ); };

							unsigned char val = ch(x);

							return make_array( val, val, val );
						}
					}
				)
        	}; //color_maps

            zen_type const& zen = static_cast<zen_type const&>( *this );

            assert( zen.row() && "save_as_bmp: matrix row cannot be zero" );
            assert( zen.col() && "save_as_bmp: matrix column cannot be zero" );

            std::string new_file_name{ file_name };
            std::string const extension{ ".bmp" };

            if ( ( new_file_name.size() < 4 )  || ( std::string{ new_file_name.begin()+static_cast<difference_type>(new_file_name.size())-4, new_file_name.end() } != extension ) )
                new_file_name += extension;

            std::ofstream stream( new_file_name.c_str(), std::ios_base::out | std::ios_base::binary );

            if ( !stream ) { return false; }

            std::string const& map_name = ( color_maps.find( color_map ) == color_maps.end() ) ? std::string{"default"} : color_map;
            //std::string const& transform_name = ( transforms.find( transform ) == transforms.end() ) ? std::string{"default"} : transform;

            auto&& selected_map = (*(color_maps.find(map_name))).second;
            auto&& selected_transform = []( double mx, double mn ) { return [=]( double v ) { return (v-mn)/(mx-mn)+1.0e-10; }; };

            unsigned char file[14] =
            {
                'B', 'M',           // magic
                0, 0, 0, 0,         // size in bytes
                0, 0,               // app data
                0, 0,               // app data
                54, 0, 0, 0         // start of data offset
            };

            unsigned char info[40] =
            {
                40, 0, 0, 0,        // info hd size
                0, 0, 0, 0,         // width
                0, 0, 0, 0,         // heigth
                1, 0,               // number color planes
                24, 0,              // bits per pixel
                0, 0, 0, 0,         // compression is none
                0, 0, 0, 0,         // image bits size
                0x13, 0x0B, 0, 0,   // horz resoluition in pixel / m
                0x13, 0x0B, 0, 0,   // vert resolutions (0x03C3 = 96 dpi, 0x0B13 = 72 dpi)
                0, 0, 0, 0,         // #colors in pallete
                0, 0, 0, 0,         // #important colors
            };

            unsigned long const the_col = zen.col();
            unsigned long const the_row = zen.row();
            unsigned long const pad_size  = ( 4 - ( ( the_col * 3 ) & 0x3 ) ) & 0x3;
            unsigned long const data_size = the_col * the_row * 3 + the_row * pad_size;
            unsigned long const all_size  = data_size + sizeof( file ) + sizeof( info );

            auto const& ul_to_uc = []( unsigned long val ){ return static_cast<unsigned char>( val & 0xffUL ); };

            file[ 2] = ul_to_uc( all_size       );
            file[ 3] = ul_to_uc( all_size >> 8  );
            file[ 4] = ul_to_uc( all_size >> 16 );
            file[ 5] = ul_to_uc( all_size >> 24 );
            info[ 4] = ul_to_uc( the_col       );
            info[ 5] = ul_to_uc( the_col >> 8  );
            info[ 6] = ul_to_uc( the_col >> 16 );
            info[ 7] = ul_to_uc( the_col >> 24 );
            info[ 8] = ul_to_uc( the_row       );
            info[ 9] = ul_to_uc( the_row >> 8  );
            info[10] = ul_to_uc( the_row >> 16 );
            info[11] = ul_to_uc( the_row >> 24 );
            info[20] = ul_to_uc( data_size       );
            info[21] = ul_to_uc( data_size >> 8  );
            info[22] = ul_to_uc( data_size >> 16 );
            info[23] = ul_to_uc( data_size >> 24 );

            stream.write( reinterpret_cast<char*>( file ), sizeof( file ) );
            stream.write( reinterpret_cast<char*>( info ), sizeof( info ) );

            unsigned char pad[3] = {0, 0, 0};
            unsigned char pixel[3];

            double const max_val = static_cast<double>( *std::max_element( zen.begin(), zen.end() ) );
            double const min_val = static_cast<double>( *std::min_element( zen.begin(), zen.end() ) );

            for ( unsigned long r = 0; r < the_row; r++ )
            {
                for ( unsigned long c = 0; c < the_col; c++ )
                {
                    auto const& rgb = selected_map( selected_transform( max_val, min_val )( zen[r][c] ) );

                    pixel[2] = rgb[0];
                    pixel[1] = rgb[1];
                    pixel[0] = rgb[2];

                    stream.write( reinterpret_cast<char*>( pixel ), 3 );
                }

                stream.write( reinterpret_cast<char*>( pad ), static_cast<std::streamsize>(pad_size) );
            }

            stream.close();

            return true;
        }

        bool save_as_pgm( std::string const& file_name ) const
        {
            zen_type const& zen = static_cast<zen_type const&>( *this );

            std::string new_file_name{ file_name };
            std::string const extension{ ".pgm" };

            if ( ( new_file_name.size() < 4 )  || ( std::string{ new_file_name.begin()+static_cast<difference_type>(new_file_name.size())-4, new_file_name.end() } != extension ) )
                new_file_name += extension;

            std::ofstream stream( new_file_name.c_str() );

            if ( !stream ) { return false; }


            //write header
            {
                stream << "P2\n";
                stream << zen.col() << " " << zen.row() << "\n";
                stream << "255\n";
                stream << "# Generated Portable GrayMap image for path [" << file_name << "]\n";
            }

            auto const& selected_map = []( double x )
            {
                assert( x >= 0.0 && "Negative x passed!" );
                assert( x <= 1.0 && "X exceeds boundary!" );

                typedef std::uint16_t  type;
                auto const& ch = []( double x_ ) { return static_cast<type>( static_cast<int>( x_ * 256.0 ) ); };
                return  ch(x);
            };

            //auto&& selected_transform = (*(transforms.find(transform_name))).second;
            auto&& selected_transform = []( double mx, double mn ) { return  [=]( double v ) { return (v-mn)/(mx-mn)+1.0e-10; }; };

            double const max_val = static_cast<double>( *std::max_element( zen.begin(), zen.end() ) );
            double const min_val = static_cast<double>( *std::min_element( zen.begin(), zen.end() ) );
            double const mmax = selected_transform( max_val, min_val )( max_val );
            double const mmin = selected_transform( max_val, min_val )( min_val );
            double const divider = mmax - mmin;

            for ( unsigned long r = 0; r < zen.row(); r++ )
            {
                for ( unsigned long c = 0; c < zen.col(); c++ )
                {
                    auto rgb = selected_map( (selected_transform( max_val, min_val )( zen[zen.row()-1-r][c] ) - mmin) / divider );
                    if ( rgb > 255.0 ) rgb = 255.0;
                    if ( rgb < 0.0 ) rgb = 0.0;

                    stream << rgb << " ";
                }
                stream << "\n";
            }

            stream.close();

            return true;
        }

        bool load( std::string const& path )
        {
            if ( path.size() < 4 )
                return load_txt( path );

            std::string extension{ path.rbegin(), path.rbegin()+4 };

            if ( extension == std::string{ "nib." } )
                return load_binary( path );

            return load_txt( path );
        }

        bool load_txt( std::string const& file_name )
        {
            zen_type& zen = static_cast<zen_type&>( *this );
            std::ifstream ifs( file_name,  std::ios::in | std::ios::binary );
            assert( ifs && "matrix::load_ascii -- failed to open file" );

			std::stringstream iss;
            std::copy( std::istreambuf_iterator<char>( ifs ), std::istreambuf_iterator<char>(), std::ostreambuf_iterator<char>( iss ) );

            std::string cache = iss.str();
            std::for_each( cache.begin(), cache.end(), [](auto& ch){ if ( ch == ',' || ch == ';' ) ch = ' '; } );
            iss.str(cache);

            std::vector<value_type> buff;
            std::copy( std::istream_iterator<value_type>( iss ), std::istream_iterator<value_type>(), std::back_inserter( buff ) );
            size_type const total_elements = buff.size();

            const std::string& stream_buff = iss.str();
            // counting the row
            size_type const r_ = static_cast<size_type>(std::count( stream_buff.begin(), stream_buff.end(), '\n' ));
            size_type const r  = *(stream_buff.rbegin()) == '\n' ? r_ : r_+1;
            size_type const c = total_elements / r;

            if ( r * c != total_elements )
            {
                std::cerr << "Error: Failed to match matrix size.\n \tthe size of matrix stored in file \"" << file_name << "\" is " << buff.size() << ".\n";
                std::cerr << " \tthe size of the destination matrix is " << r << " by " << c << " elements.\n";
                std::cerr << " \tthe buff is:\n";
                std::copy( buff.begin(), buff.end(), std::ostream_iterator<value_type>( std::cerr, "\t" ) );
                std::cerr << "\n";
                std::cerr << " \tthe stream_buffer is:\n" << stream_buff << "\n";
                return false;
            }

            zen.resize( r, c );
            std::copy( buff.begin(), buff.end(), zen.begin() );
            ifs.close();

            return true;
        }

        bool load_binary( std::string const& file_name )
        {
            zen_type& zen = static_cast<zen_type&>( *this );

            std::ifstream ifs( file_name, std::ios::binary );
            assert( ifs && "matrix::load_binary -- failed to open file" );
            if ( !ifs ) return false;

            std::vector<char> buffer{(std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>())};

            assert( buffer.size() >= sizeof(size_type) + sizeof(size_type) && "matrix::load_library -- file too small, must be damaged" );
            if ( buffer.size() <= sizeof(size_type) + sizeof(size_type) ) return false;

            size_type r;
            std::copy( buffer.begin(), buffer.begin()+sizeof(r), reinterpret_cast<char*>( std::addressof( r ) ) );

            size_type c;
            std::copy( buffer.begin()+sizeof(r), buffer.begin()+sizeof(r)+sizeof(c), reinterpret_cast<char*>( std::addressof( c ) ) );

            zen.resize( r, c );
            assert( buffer.size() == sizeof(r)+sizeof(c)+sizeof(Type)*zen.size() && "matrix::load_binary -- data does not match, file damaged" );
            if ( buffer.size() != sizeof(r)+sizeof(c)+sizeof(Type)*zen.size() ) return false;

            std::copy( buffer.begin()+sizeof(r)+sizeof(c), buffer.end(), reinterpret_cast<char*>(zen.data()) );

            return true;
        }

#if 0
        bool load_bmp( std::string const& file_name )
        {
        }

        bool load_pgm( std::string const& file_name )
        {
        }
#endif


    };//struct

}

#endif//RBAXERJBEYLCERDFQTJFOVMUUESBKJJLCNETALSFLEMTXSHYBEWUIBSGYLSKQXIYNHPJSJXRS

