#include "Settings.h"
#include <string>



Settings::Settings()
{
}


Settings::~Settings()
{
}




void Settings::LoadStreamFile(char* Filename,char* DatasetFIntensity, bool InclMultiHits)
{

	Echo("Start reading stream-file.");

	using std::getline;
	using std::stringstream;

	std::vector<HitEvent> tmpHitEvents;

	std::ifstream File(Filename);
	HitEvent currEvent;
	for (std::string line; getline(File, line);)
	{


		if (line.compare(0, 15, "Image filename:") == 0)
		{
			currEvent = HitEvent();
			stringstream ss(line);
			ss.seekg(16);
			ss >> currEvent.Filename;
			continue;
		}
		if (line.compare(0, 20, "Image serial number:") == 0)
		{
			stringstream ss(line);
			ss.seekg(21);
			ss >> currEvent.SerialNumber;
			continue;
		}
		if (line.compare(0, 6, "Event:") == 0)
		{
			stringstream ss(line);
			ss.seekg(9);
			ss >> currEvent.Event;
			continue;
		}
		if (line.compare("--- Begin crystal") == 0)
		{
			//shots.push_back(CRYSTAL());

			while (getline(File, line))
			{
				if (line.find("Cell parameters") == 0) break;
			}
			for (size_t i = 0; getline(File, line) && i<3; ++i) {
				double x, y, z;
				stringstream ss(line);
				ss.seekg(7);
				ss >> x >> y >> z;
				//         cout << line << endl;
				currEvent.RotMatrix[i * 3 + 0] = x;
				currEvent.RotMatrix[i * 3 + 1] = y;
				currEvent.RotMatrix[i * 3 + 2] = z;
			}
			currEvent.Dataset = DatasetFIntensity;
			tmpHitEvents.push_back(currEvent);

		}
	}

	if (!InclMultiHits)//check for double or multiple crystals in one hit and ignore them
	{
		Echo("Remove multiple hits.");
		//check first event:
		if (!(tmpHitEvents[0].Event == tmpHitEvents[1].Event &&  tmpHitEvents[0].Filename == tmpHitEvents[1].Filename))
			HitEvents.push_back(tmpHitEvents[0]);
		//check events [1:end-1]
		for (int i = 1; i < tmpHitEvents.size() - 1; i++)
		{
			if ((tmpHitEvents[i].Event == tmpHitEvents[i - 1].Event &&  tmpHitEvents[i].Filename == tmpHitEvents[i - 1].Filename) || (tmpHitEvents[i].Event == tmpHitEvents[i + 1].Event && tmpHitEvents[i].Filename == tmpHitEvents[i + 1].Filename))
				continue;
			HitEvents.push_back(tmpHitEvents[i]);
		}
		//check last event
		if (!(tmpHitEvents[tmpHitEvents.size() - 1].Event == tmpHitEvents[tmpHitEvents.size() - 2].Event &&  tmpHitEvents[tmpHitEvents.size() - 1].Filename == tmpHitEvents[tmpHitEvents.size() - 2].Filename))
			HitEvents.push_back(tmpHitEvents[tmpHitEvents.size() - 1]);
		if (echo)
			std::cout << HitEvents.size() << " single hits within " << tmpHitEvents.size() << " total hits loaded.\n";
	}
	else
	{
		HitEvents = tmpHitEvents;
		if (echo)
			std::cout << HitEvents.size() << " total hits loaded.\n";
	}
	Echo("Done reading stream-file.\n");
}



void Settings::Echo(std::string output)
{
	if (echo)
		std::cout << output << "\n";

}
