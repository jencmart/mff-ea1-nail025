## TSP1
* Máte zase několik možností, co zkusit:
  * KRIZENI:
    * PMX ... partially mapped crossover [ne]
      OX  ... ordered crossover [je]
      CX  ... cyclic recomb  [ne]
      ER  ... edge rocombination [DONE]
  * MUTACE:
    * Swap (1+ elems)           ... [je]
    * 2OPT (3OPT)               ... [DONE] ... 2opt je vlastne dani nake casti pozpatku..
        * for each dvojice hran. zkus udelat jine spojeni
        * A ---- B   C ---- D  ...... ZKUSIME ::  A-D, C-B  OR A-C, B-D
  * INICIALIZACE:
    * Nerest Neighbours ...
    * Edge insertion algorithm ... add 1 node ... +2 edges, -1 edge
    * Nejmensi kostra ... 
  * upravit fitness, (1/délka cesty nemusí být nejlepší, ale jestli používáte turnaj, tak je to jedno...

*                  best == 158 418 km
[Bonus] tsp_std.in cestu < 170 000 km 
[Bonus] tsp_std.in cestu < 160 000 km

## TSP 2
Zkuste použít některou z heuristik
kterou jste nepoužili minule a zlepšit svoje řešení
pošlete mi porovnání minulého a nového řešení

