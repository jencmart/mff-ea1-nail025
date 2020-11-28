Pravidla pro klasifikaci. 

rules.py
jednoduchou implementaci inspirovanou Pittsburghským přístupem,
tj. jedinci jsou množiny několika pravidel. 
Naším cílem je klasifikovat správně daný dataset,
tedy používáme fitness funkci,
která počítá správnost klasifikace (procento správně klasifikovaných instancí).
Jako objective funkci používáme 1 - správnost.

Jedinci jsou v našem případě seznamy pravidel (maximální počet pravidel lze nastavit),
každé pravidlo se skládá z podmínek pro každý atribut.
Máme tři druhy podmínek - menší než, větší než a univerzální podmínku, která je vždy splněna.
Při vyhodnocení jedince se na každé instanci z dat kontroluje,
která pravidla jí odpovídají (jsou splněny všechny podmínky)
a tato pravidla se nechají hlasovat o klasifikaci (tj. vyhraje třída, kterou pravidla předpovídají nejčastěji).

Pro tyto jedince máme implementované tři genetické operátory
    křížení, které připomíná uniformní křížení (berou se náhodně pravidla z jednoho nebo druhého jedince)
    mutace, která mění hranice v pravidlech
    mutace, která mění hodnotu předpovídané třídy


Testovaci data
- iris.csv - y=druh kostace(1-3) ;;  x=rozmery okvětních lístků
------------------------------------------------------

Vylepsete algoritmus vyzkoušejte několik z nich (nebo si vymyslete i vlastní):
Přidat další typy podmínek.
Přidat další genetické operátory.
Přidat váhy/priority k jednotlivým pravidlům.
Změnit parametry algoritmu (počet pravidel v jedinci, nastavení operátorů apod.)


Porovnejte vaše výsledky s implementací ve zdrojových kódech na obou datasetech. Nezapomeňte připojit grafy s průběhem.
