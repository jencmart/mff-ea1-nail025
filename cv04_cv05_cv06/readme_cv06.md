Pokud chceme použít lamarkistické nebo baldwinistické myšlenky ve spojité optimalizaci 
udelame to jako uceni behem zivota
lamarckismus = v rámci mutace [ jedince realne zmenime ]
baldwinismus = v rámci výpočtu fitness [ + jedince realne nezmenime ] 

uceni == vylepseni --> odecist gradient (nebo nejaky jiny trapny shit jako simulovane zihani..)
(numericalDerivative v co_functions.py) 
potom: jedinec - lr* numericalDerivative(fitness(jedinec))  (lr=0.1)



Zdrojáky mezapočítávají tedy vyhodnocení, které potřebujete pří lokálním prohledávání
( jen pocet generaci a velikost populace )
Zohledněte tento fakt při komentování výsledků.
výpočet gradientu == gradient se vola |jedinec|-krat... 

Porovnejte Lamarcka i Baldwina se svými dřívějšími přístupy a napište mi, na co jste přišli.
