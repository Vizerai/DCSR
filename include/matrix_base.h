#ifndef MATRIX_BASE_H
#define MATRIX_BASE_H

template<typename IndexType, typename MemorySpace>
class binary_vector_base
{
	public:
		size_t num_rows;		//column vector
        size_t num_entries;

		binary_vector_base()
			: num_rows(0), num_entries(0) {}

		template <typename Matrix>
        binary_vector_base(const Matrix& m)
            : num_rows(m.num_rows), num_cols(m.num_cols), num_entries(m.num_entries) {}

        binary_vector_base(size_t rows)
            : num_rows(rows), num_entries(0) {}

        binary_vector_base(size_t rows, size_t entries)
            : num_rows(rows), num_entries(entries) {}

		void resize(size_t rows, size_t entries)
        {
            num_rows = rows;
            num_entries = entries;
        }

        void swap(binary_matrix_base& base)
        {
        	thrust::swap(num_rows,    base.num_rows);
        	thrust::swap(num_entries, base.num_entries);
		}
};

#endif