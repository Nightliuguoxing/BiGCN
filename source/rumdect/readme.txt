- Twitter Data
* Twitter.txt: This corpus contains 992 labeled events in total. Each line contains one event with the ids of relevant tweets: event_id, label, tweet_ids. For the labels, the value is 1 if the event is a rumor, and is 0 otherwise. Note that we cannot release the specific content of tweets due to the terms of use of Twitter data. Users can download the content themselves via Twitter API.
 
* Twitter_event_claims.txt: This file provides the content of the main claim of each event. Each line contains one event with its claim consisting of event_id and claim content. In this file, only the claims of the events collected from Snopes.com by ourselves are included, and the claims marked as 'unknown' are borrowed from other datasets (for balancing our class distribution) constructed in the following two papers:

  ** @inproceedings{castillo2011information,
       title={Information credibility on twitter},
       author={Castillo, Carlos and Mendoza, Marcelo and Poblete, Barbara},
       booktitle={Proceedings of the 20th international conference on World wide web},
       pages={675--684},
       year={2011},
       organization={ACM}
    } 
  ** @inproceedings{kwon2013prominent,
       title={Prominent features of rumor propagation in online social media},
       author={Kwon, Sejeong and Cha, Meeyoung and Jung, Kyomin and Chen, Wei and Wang, Yajun},
       booktitle={2013 IEEE 13th International Conference on Data Mining},
       pages={1103--1108},
       year={2013},
       organization={IEEE}
    }

- Weibo Data (Weibo.txt): This corpus contains 4664 labeled events in total. Each line contains one event with ids of relevent posts with format: event_id, label, post_ids. For the labels, the value is 1 if the event is a rumor, and is 0 otherwise. We also release the content of all the posts in json format which are saved in the ./Weibo directory, where each file is named as event_id.json, corresponding to individual event.
  ** Note that we do not explicitly produce claim for each Weibo event since we directly collect the events as propagation threads from Sina community management center, and the source post (the first post) of each event can be considered as the main claim.
