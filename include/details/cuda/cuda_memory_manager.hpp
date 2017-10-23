#ifndef GFLWDOFFXMAENDLYOSYUKEWWYLHKIARBXEGVGHILVEYHHWHRIFCTXDXEGHQGHXWMPVFGVNQWK
#define GFLWDOFFXMAENDLYOSYUKEWWYLHKIARBXEGVGHILVEYHHWHRIFCTXDXEGHQGHXWMPVFGVNQWK

#include "./cuda_memory_cache.hpp"
#include "../device_matrix_view/device_matrix_view.hpp"

namespace cunn
{

    static cuda_memory_cache    cmc;

    namespace details
    {

        std::vector<device_matrix_view<float>*> to_pointers( device_matrix_view<float>* dmv )
        {
            std::vector<device_matrix_view<float>*> ans;
            ans.emplace_back( dmv );
            return ans;
        }

        template<typename Itor >
        std::vector<device_matrix_view<float>*> to_pointers( std::tuple<Itor,Itor> const& tp )
        {
            auto [begin,end] = tp;
            std::vector<device_matrix_view<float>*> ans;
            while ( begin != end )
                ans.emplace_back( std::addressof( *begin++ ) );
            return ans;
        }

        template< typename T, typename ... Ts >
        std::vector<device_matrix_view<float>*> to_pointers( T t, Ts ... ts )
        {
            std::vector<device_matrix_view<float>*> t1 = to_pointers( t );
            std::vector<device_matrix_view<float>*> t2 = to_pointers( ts... );

            t2.insert( t2.begin(), t1.begin(), t1.end() );

            return t2;
        }

    }//namespace

    struct cuda_memory_manager
    {
        cuda_memory_manager( int i = -1, float* a = nullptr, unsigned long s = 0 ) : id_( i ), address_(a), size_(s) {}

        int                 id_;
        float*              address_;
        unsigned long       size_;

        template< typename ... Views >
        void manage( Views ... views )
        {
            assert( address_ == nullptr && size_ == 0UL && id_ == -1 && "Error: manager cannot mange more GPU instances." );

            std::vector<device_matrix_view<float>*> view_references = details::to_pointers( views... );

            for ( auto& m : view_references )
                size_ += (*m).size();

            auto [gpu_id, address] = cmc.cache( size_ * sizeof(float) );

            id_ = gpu_id;
            address_ = reinterpret_cast<float*>( address );

            float* current_address = address_;
            for ( auto& m : view_references )
            {
                device_matrix_view<float> tmp{ (*m).row(), (*m).col(), current_address };
                (*m).swap( tmp );
                current_address += (*m).size();
            }
        }

        ~cuda_memory_manager()
        {
            if ( address_ )
                cmc.decache( id_, reinterpret_cast<void*>(address_), size_*sizeof(float) );

            id_ = -1;
            address_ = nullptr;
            size_ = 0UL;
        }

    };

}//namespace cunn

#endif//GFLWDOFFXMAENDLYOSYUKEWWYLHKIARBXEGVGHILVEYHHWHRIFCTXDXEGHQGHXWMPVFGVNQWK

