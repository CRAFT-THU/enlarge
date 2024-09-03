
#ifndef HELPER_CPP
#define HELPER_CPP

template<typename T>
void del_content_vec(T &vec)
{
	for (auto iter = vec.begin(); iter != vec.end(); iter++) {
		if (!(*iter)) {
			continue;
		}
		delete (*iter);
	}
	vec.clear();
}

template<typename T>
void del_content_map(T &map)
{
	for (auto iter = map.begin(); iter != map.end();  iter++) {
		if (!(iter->second)) {
			continue;
		}
		delete (iter->second);
	}
	map.clear();
}

#endif /* HELPER_CPP */
