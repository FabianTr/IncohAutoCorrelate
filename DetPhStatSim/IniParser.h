#pragma once
#include <iostream>
#include <string>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <algorithm> 
#include <cctype>
#include <locale>
#include <vector>

class IniParser
{
	std::unordered_map<std::string,std::string> Dictionary;

	

	// trim from start (in place)
	inline void ltrim(std::string& s) {
		s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
			return !std::isspace(ch);
			}));
	}

	// trim from end (in place)
	inline void rtrim(std::string& s) {
		s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
			return !std::isspace(ch);
			}).base(), s.end());
	}

	// trim from both ends (in place)
	inline void trim(std::string& s) {
		ltrim(s);
		rtrim(s);
	}

	template <typename T>
	T Convert(const std::string& Input)
	{
		std::stringstream ss(Input);
		T ret;
		ss >> ret;
		return ret;
	}
public:

	static std::vector<std::string>StringSplitter(std::string Input, std::string delimiter)
	{
		//Splits "delimiter"-seperated Values within string to vector
		std::vector<std::string> Output;
		size_t pos = 0;
		std::string token;
		//bool ParsingAlive = true;
		while ((pos = Input.find(delimiter)) != std::string::npos)
		{
			token = Input.substr(0, pos);
			Output.push_back(token);
			Input.erase(0, pos + delimiter.length());
		}
		if (Input.size() != 0)
			Output.push_back(Input);
		return Output;
	}

	static inline bool FileExists(const std::string& Filename) 
	{
		std::ifstream f(Filename.c_str());
		return f.good();
	}


	void Clear()
	{
		Dictionary.clear();
	}
	bool LoadFile(std::string Filename,  std::string delimiter = "=")
	{
		if (FileExists(Filename))
		{
			std::ifstream file(Filename);
			std::string line;
			while (std::getline(file, line))
			{
				trim(line);
				if (line.length() == 0) //check for empty line
					continue;
				if (line[0] == '#') //check for comment
					continue;

				std::string Key = line.substr(0, line.find(delimiter));
				line.erase(0, line.find(delimiter) + delimiter.length());
				std::string Val = line;
				trim(Key);
				trim(Val);
				Dictionary[Key] = Val;
			}
			return true;
		}
		else
		{
			return false;
		}

	}

	void SafeFile(std::string Filename)
	{
		std::ofstream file(Filename);
		for (auto& it : Dictionary)
		{
			file << it.first << '=' << it.second << "\n";
		}
		file.close();
	}


	template <typename T> 
	T GetValue(std::string Key, T Default = T{})
	{
		if (Dictionary.count(Key) == 0)
			return Default;
		else
		{
			T ret = Convert<T>(Dictionary[Key]);
			return ret;
		}
	}

	template <typename T>
	void SetValue(std::string Key, T Input)
	{
		std::stringstream ss;
		ss << Input;

		Dictionary[Key] = ss.str();
	}

	void SetValueStr(std::string Key_Input, std::string delimiter = "=")
	{
		trim(Key_Input);
		if (Key_Input.length() == 0) //check for empty line
			return;

		std::string Key = Key_Input.substr(0, Key_Input.find(delimiter));
		Key_Input.erase(0, Key_Input.find(delimiter) + delimiter.length());
		std::string Val = Key_Input;
		trim(Key);
		trim(Val);
		Dictionary[Key] = Val;
	}

};

