#ifndef CUDA_MEMORY_CACHE_HPP_INCLUDED_SDPJ4LKJA4OIJSDFLKJ498JISAFLKJF98I4KLJSDF
#define CUDA_MEMORY_CACHE_HPP_INCLUDED_SDPJ4LKJA4OIJSDFLKJ498JISAFLKJF98I4KLJSDF

#include "../utility.hpp"
#include "./cuda.hpp"

namespace cunn
{

    struct allocated_memory
    {
        unsigned long       size_in_byte_;
        void*               address_;
    };

    struct cuda_memory_cache
    {
        std::map<int, unsigned long>                available_;
        std::multimap<int, allocated_memory>        allocated_;

        ~cuda_memory_cache() //TODO
        {
        }

        cuda_memory_cache()
        {
            int gpu_numbers = get_devices();
            assert( gpu_numbers && "No GPU found." );

            for ( int idx = 0; idx != gpu_numbers; ++idx ) // device id start from 0
            {
                int current_device_capability = capability( idx ); //max memory in bytes

                if ( current_device_capability < 300 ) continue;

                available_[idx] = global_memory(idx);
            }

            if ( !available_.size() )
                assert( !"No usable GPU found." );
        }

        auto cache( unsigned long memory_to_cache_in_byte )
        {
            //find GPU with max memory
            auto max_itor = available_.begin();
            {
                for ( auto itor = max_itor; itor != available_.end(); ++itor )
                {
                    if ( (*itor).second > (*max_itor).second )
                        max_itor = itor;
                }
            }

            //allocate memory within this GPU
            auto& [gpu_id, size_in_byte] = *max_itor;
            assert( size_in_byte >= memory_to_cache_in_byte && "Not enough memory" );
            set( gpu_id );
            void* allocated_memory_address = reinterpret_cast<void*>(allocate<unsigned char>( memory_to_cache_in_byte ));

            //decrease available_ record
            size_in_byte -= memory_to_cache_in_byte;

            //increase allocated_ record
            allocated_.emplace( std::make_pair( gpu_id, allocated_memory{ memory_to_cache_in_byte, allocated_memory_address } ) );

            //return tuple
            return std::make_tuple( gpu_id, allocated_memory_address );
        }

        bool decache( int gpu_id, void* address_to_decache, unsigned long memory_to_decache_in_byte )
        {
            assert( available_.find(gpu_id) != available_.end() && "No available record found" );
            assert( allocated_.find(gpu_id) != allocated_.end() && "No allocation record found" );
            //decrease allocated_
            for ( auto itor = allocated_.begin(); itor != allocated_.end(); ++itor )
            {
                //auto const&[id, address, memory_size] = *itor;
                auto const&[id, allocated_memory_record] = *itor;
                auto const&[memory_size, address] = allocated_memory_record;
                if ( id == gpu_id && address == address_to_decache && memory_size == memory_to_decache_in_byte )
                {
                    //deallocate
                    set( gpu_id );
                    deallocate( address_to_decache );
                    //increase available_
                    available_[gpu_id] += memory_to_decache_in_byte;
                    return true;
                }
            }

            return false;
        }

    };

}//namespace

#endif//CUDA_MEMORY_CACHE_HPP_INCLUDED_SDPJ4LKJA4OIJSDFLKJ498JISAFLKJF98I4KLJSDF
