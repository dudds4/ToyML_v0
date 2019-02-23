#ifndef NODESET_H
#define NODESET_H

#include "graph.h"
#include <memory>

template<typename NodeT>
struct NodeSet
{
	NodeSet(unsigned size)
	: m_size(size)
	{
		if(!size)
			throw new std::exception();

		inputs = std::shared_ptr<NodeT[]>(new NodeT[size]);
	}

	NodeSet(const std::vector<Node*> &parents)
	: m_size(parents.size())
	{
		if(!m_size)
			throw new std::exception();

		inputs = std::shared_ptr<NodeT[]>(new NodeT[m_size]);
		
		for(unsigned i = 0; i < m_size; ++i)
		{
			inputs[i].setParents({parents.at(i)});
		}
	}

	NodeT& at(size_t index)
	{
		if(index > m_size)
			throw new std::exception();

		return inputs[index];
	}

	NodeT* ptrAt(size_t index)
	{
		if(index > m_size)
			throw new std::exception();

		auto ptr = inputs.get();
		return (ptr + index);		
	}

	std::vector<Node*> getNodes()
	{
		std::vector<Node*> r;
		r.reserve(m_size);
		
		auto ptr = inputs.get();
		for(unsigned i = 0; i < m_size; ++i)
			r.push_back(ptr+i);

		return r;
	}

	std::vector<Node*> getNodes(int l, int h)
	{
		std::vector<Node*> r;
		r.reserve(m_size);
		
		auto ptr = inputs.get();
		for(unsigned i = l; i < m_size && i < h; ++i)
			r.push_back(ptr+i);

		return r;
	}

	std::vector<NodeT*> getInputs()
	{
		std::vector<NodeT*> r;
		r.reserve(m_size);
		
		auto ptr = inputs.get();
		for(unsigned i = 0; i < m_size; ++i)
			r.push_back(ptr + i);

		return r;
	}

private:
	std::shared_ptr<NodeT[]> inputs;
	unsigned m_size;
};

#endif