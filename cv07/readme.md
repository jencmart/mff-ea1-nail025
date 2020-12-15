fitness = accuracy
bjective = 1-accuracy (minimalizujeme)

Jedinci jsou v našem případě seznamy pravidel, každé pravidlo se skládá z podmínek pro každý atribut.
Máme tři druhy podmínek -  <, >, any

vyhodnoceni: vyberou se matchujici pravidla, vyhraje ta trida, ktera je nejcasteji (v nejvice matchujicich pravidlech)

Mame implementovane 3 gen.op.:
    křížení - "uniformní křížení"
    mutace - mění hranice v pravidlech (meni C v pravidlu :  x {<,>,any} C)
    mutace -  mění hodnotu předpovídané třídy (meni y)


Testovaci data
- iris.csv - y=druh kostace(1-3) ;;  x=rozmery okvětních lístků
------------------------------------------------------

Vylepsete algoritmus:
    Přidat další typy podmínek:
        * condition between [OK]
    Přidat další genetické operátory:
        * mutace - zmenit condition [OK]
        * mutace - weight rule mutate [OK]
    Přidat váhy/priority k jednotlivým pravidlům:
        * pridat wahu pro rule [OK]
    Změnit parametry algoritmu:
        * počet pravidel v jedinci [todo-grid?]
        * nastavení operátorů      [todo-grid?]

Porovnejte vaše výsledky s implementací ve zdrojových kódech na obou datasetech.
Nezapomeňte připojit grafy s průběhem.
