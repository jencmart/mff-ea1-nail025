Diferencialni Evoluce
 
 * MUTACE = k danému jedinci přičítá rozdíl jiných dvou jedinců náhodně vybraných z populace vynásobený parametrem F.
 * KRIZENI = vyberedalší náhodný jedinec (rodič) z populace, ten se prochází zároveň s výsledkem mutace a s pravděpodobností 1-CR 
              se do zmutovaného jedince přenese číslo ze stejné pozice v rodiči. 
              CR je pravděpodobnost, že se pozice ve zmutovaném jedinci nezmění)
              Vždy se přenese alespoň jedna hodnota.
 * SELEKCE =  při té se porovná původní jedinec (před mutací)
              s jedincem,  který vznikne po křížení 
              a lepší z těchto dvou přežívá do další generace.

Typické F=0.8 a CR=0.9.  (Pro separabilni fce cast CR=0.2 ... 80% se prenese z rodice)


Zkuste si naimplementovat vlastní operátory inspirované diferenciální evolucí a porovnejte je s operátory z minula. 
  1. Zkuste udělat přímo diferenciální evoluci
  2. Zkuste v diferenciální mutaci použít více než dva jedince, ze kterých se počítá rozdíl
  3. Zkuste měnit parametry F a CR nějak adaptivně