---
title: "STA102 : régression linéaire simple"
---

Chargement du jeu de données :

```{r }
df <- read.delim("D:/Etudes/2025_2026/STA102/Regression_simple/df.txt")
```


### Contexte : 

Soit $Y$ une variable **quantitative** (à expliquer) et $X$ une variable explicative :

$X_{p} \ : \ p \in [1,...,p]$

Et le vecteur d'observations composé des couples :

$(x_{i} \ ; \ y_{i})$

### Modèle de régression simple : 

$$Y=X\beta+\epsilon$$

Représentation graphique :

L'affichage du nuage de points donne une première impression sur l'existence d'une corrélation entre $X$ et $Y$ :

```{r Affchage du nuage de points}
# Graphique :
plot(df[2:3])
```

Quantification de la force de la relation linéaire entre $X$ et $Y$ :

$$r=\frac{\frac{1}{n}\Sigma^{n}_{i=1}(x_{i}-\bar{x})(y_{i}-\bar{y})}{\sqrt{\frac{1}{n}\Sigma^{n}_{i=1}(x_{i}-\bar{x})^{2} \ \frac{1}{n}\Sigma^{n}_{i=1}(y_{i}-\bar{y})^{2}}}$$

$$=\frac{\Sigma^{n}_{i=1}(x_{i}-\bar{x})(y_{i}-\bar{y})}{\sqrt{\Sigma^{n}_{i=1}(x_{i}-\bar{x})^{2} \ \Sigma^{n}_{i=1}(y_{i}-\bar{y})^{2}}}$$

$$=\frac{\Sigma^{n}_{i=1}x_{i}y_{i} \ \  n\bar{x}\bar{y}} {(\sqrt{\Sigma^{n}_{i=1}x_{i}^{2}-n\bar{x}^{2}) \ (\Sigma^{n}_{i=1}y_{i}^{2}-n\bar{y}^{2})}}=\frac{S_{XY}}{S_{X}S_{Y}}$$

Voir théorème de **König-Huygens**.      

Propriétés : 

- $r\in[-1,1]$

**Attention :**

- $r=0 \ \nRightarrow X \perp Y$

### Principe des moindres carrés

On cherche `a approximer le nuage des points

$(x_{i}, y_{i})$ par une droite d’´equation $\hat{y} = a+bx$ de telle sorte que

$\Sigma^{n}_{i=1}(y_{i}− \hat{y}_{i})$ soit minimale :

$$e_{i}=y_{i}-\hat{y_{i}}$$

Avec :

$$Min(\Sigma^{n}_{i=1}y_{i}-\hat{y_{i}})^{2}$$

### Détermination des coefficients : 

Soit : 

$F(a;b)=(\Sigma^{n}_{i=1}y_{i}-a -by_{i})^{2}$

$$\frac{\partial \ F(a;b)}{\partial \ a}=2\Sigma^{n}_{i=1}(y_{i}-a -bx_{i})=0$$

$$\Rightarrow \ \Sigma^{n}_{i=1}e_{i}=0$$

$$\frac{\partial \ F(a;b)}{\partial \ b}=2\Sigma^{n}_{i=1}x_{i}(y_{i}-a -bx_{i})=0$$

$$\Rightarrow \ \Sigma^{n}_{i=1}x_{i}e_{i}=0$$

Donc : 

- $E[\ e\ ]=0$
- $Cov(x,e)=0$

Résolution : 

$$\Sigma^{n}_{i=1}(y_{i}-a -bx_{i})=\Sigma^{n}_{i=1}y_{i} -na -\Sigma^{n}_{i=1}bx_{i}=0$$

$$\Rightarrow \ \frac{\Sigma^{n}_{i=1}y_{i}}{n} -a -b\frac{\Sigma^{n}_{i=1}x_{i}}{n}= \bar{y}-a-b\bar{x}= 0$$

$$\Rightarrow \ a=\bar{y}-b\bar{x}$$

et :

$$\Sigma^{n}_{i=1}x_{i}(y_{i}-a -bx_{i})=\Sigma^{n}_{i=1}x_{i}(y_{i}-\bar{y}+b\bar{x} -bx_{i})=\Sigma^{n}_{i=1}x_{i}(y_{i}-\bar{y})+b\bar{x}\Sigma^{n}_{i=1}x_{i}-b\Sigma^{n}_{i=1}x_{i}^{2}= 0$$

$$\Sigma^{n}_{i=1}x_{i}(y_{i}-\bar{y})+bn\bar{x}^{2}-b\Sigma^{n}_{i=1}x_{i}^{2}= 0$$

$$bn\bar{x}^{2}-b\Sigma^{n}_{i=1}x_{i}^{2}= \Sigma^{n}_{i=1}x_{i}(y_{i}-\bar{y})$$

$$b(\Sigma^{n}_{i=1}x_{i}^{2}-n\bar{x}^{2})= \Sigma^{n}_{i=1}x_{i}(y_{i}-\bar{y})$$

$$b= \frac{\Sigma^{n}_{i=1}x_{i}(y_{i}-\bar{y})}{\Sigma^{n}_{i=1}x_{i}^{2}-n\bar{x}^{2}}$$

Avec :

$$
x_{i}=x_{i}+\bar{x}-\bar{x} \ \Rightarrow \ 
\sum_{i=1}^n x_i (y_i-\bar y)
= \sum_{i=1}^n \big[(x_i-\bar x)+\bar x\big](y_i-\bar y)
$$

$$
= \sum_{i=1}^n (x_i-\bar x)(y_i-\bar y)
+ \bar x \sum_{i=1}^n (y_i-\bar y)
$$

et :

$$
\sum_{i=1}^n (y_i-\bar y)=0 \ \Rightarrow \ \sum_{i=1}^n x_i (y_i-\bar y)
= \sum_{i=1}^n (x_i-\bar x)(y_i-\bar y)
$$

$$
(x_i-\bar x)^2 = x_i^2 - 2\bar x x_i + \bar x^2
\ \Rightarrow \ 
\sum_{i=1}^n (x_i-\bar x)^2
= \sum_{i=1}^n x_i^2
- 2\bar x \sum_{i=1}^n x_i
+ n\bar x^2
$$

$$
\sum_{i=1}^n x_i = n\bar x
\ \Rightarrow \ 
\sum_{i=1}^n (x_i-\bar x)^2
= \sum_{i=1}^n x_i^2 - n\bar x^2
$$

Donc :

$$
b= \frac{\sum_{i=1}^n x_i(y_i-\bar y)}
       {\sum_{i=1}^n x_i^2 - n\bar x^2}
= \frac{\sum_{i=1}^n (x_i-\bar x)(y_i-\bar y)}
       {\sum_{i=1}^n (x_i-\bar x)^2}
$$

On fais le lien avec le coefficient de corrélation : 

$$
r = \frac{s_{XY}}{s_X s_Y} \ \Longrightarrow \ 
s_{XY} = r \, s_X s_Y \ \Longrightarrow \
\frac{s_{XY}}{s_X^2} = \frac{r \, s_X s_Y}{s_X^2} \ \Longrightarrow \ 
\frac{s_{XY}}{s_X^2} = r \, \frac{s_Y}{s_X}
$$

$$\Rightarrow \ b=\frac{\Sigma^{n}_{i=1}x_{i}y_{i} \ \  n\bar{x}\bar{y}} {\sum_{i=1}^n x_{i}^{2}-n\bar x^{2}}=\frac{S_{XY}}{S_{X}^{2}}$$

#### Equation de la droite des moindres carrés :

$$\hat{y}=a+bx=\bar{y}-b\bar{x}+r\frac{S_{Y}}{S_{X}}x=\bar{y}-r\frac{S_{Y}}{S_{X}}\bar{x}+r\frac{S_{Y}}{S_{X}}x=\bar{y}+r\frac{S_{Y}}{S_{X}}(x-\bar{x})$$

Propriétés : 

- La droite des moindres carrés passe par $(\bar{x} \ ; \ \bar{y})$, le centre de gravité du nuage.       
- Son signe est le même que celui de $S_{XY}$.

### Contribution de chaque observation :

**Rappel :** la pente ($b$) de la droite est calculée sur les observations $(x_{i} \ ; \ y_{i})$.

$$b= \frac{\sum_{i=1}^n (x_i-\bar x)(y_i-\bar y)}
       {\sum_{i=1}^n (x_i-\bar x)^2}
  =\sum^{n}_{i=1}\left[\frac{(x_{i}-\bar{x})^{2}}
  {\Sigma^{2}_{i=1}(x_{i}-\bar{x})^{2}}\right]\frac{(y_{i}-\bar{y})}{(x_{i}-\bar{x})}$$

La pente de la droite des moindres carrées peut être interprétée comme **la moyenne pondérée des pentes des droites passant par le barycentre $(\bar{x}, \bar{y})$ du nuage des points et chaque point-observation $(x_{i}, y_{i})$.**

#### Levier : 

$$h_{i}=\frac{1}{n}+\frac{(x_{i}-\bar{x})^{2}}
  {\Sigma^{2}_{i=1}(x_{i}-\bar{x})^{2}}$$

Propriétés : 

- $h_{i}\in[\frac{1}{n}\ ;\ 1]$       
- $h_{i}=1 \ \Rightarrow \ \{x_{i}=x_{j} \ ; \ \forall \ i \neq j\}$      
- $h_{i}=\frac{1}{n} \ \Rightarrow \ \{x_{j}=x_{k} \ ; \ \forall \ (j \neq k) \neq i\}$
- Généralement, un levier suppérieur à $4/n$ est considéré comme important

**Attention :** le levier décrit un **potentiel**. Il ne décrit pas **directement** l'influence de l'observation. 

### Résidus et qualité d'ajustement : 

#### Décomposition de la variation totale :

$$SCT=\sum^{n}_{i=1}(y_{i}-\bar{y}_{i})^{2}=\sum^{n}_{i=1}(y_{i}-\hat{y}_{i}+\hat{y}_{i}-\bar{y}_{i})^{2}= \\ 
\sum^{n}_{i=1}(y_{i}-\hat{y}_{i})^{2}-2\sum^{n}_{i=1}(y_{i}-\hat{y}_{i})(\hat{y}_{i}-\bar{y}_{i})+\sum^{n}_{i=1}(y_{i}(\hat{y}_{i}-\bar{y}_{i})^{2} \\
$$

Avec :

$$
2\sum^{n}_{i=1}(y_{i}-\hat{y}_{i})(\hat{y}_{i}-\bar{y}_{i})=2\sum^{n}_{i=1}(y_{i}-\hat{y}_{i})\hat{y}_{i}-2\sum^{n}_{i=1}(y_{i}-\hat{y}_{i})\bar{y}_{i}
$$

Où :

$$
=2\sum^{n}_{i=1}(y_{i}-\hat{y}_{i})-2\bar{y}_{i}\sum^{n}_{i=1}(y_{i}-\hat{y}_{i})=2\sum^{n}_{i=1}(y_{i}-\hat{y}_{i})-2\bar{y}_{i}\sum^{n}_{i=1}e_{i}=2\sum^{n}_{i=1}(y_{i}-\hat{y}_{i}) \\
=2\sum^{n}_{i=1}e_{i}(a+bx_{i})=2a\sum^{n}_{i=1}e_{i}+b2\sum^{n}_{i=1}e_{i}x_{i}=0
$$

$$SCT=\sum^{n}_{i=1}(y_{i}-\bar{y}_{i})^{2}=\sum^{n}_{i=1}(y_{i}-\hat{y}_{i})^{2}+\sum^{n}_{i=1}(\hat{y}_{i}-\bar{y}_{i})^{2} \\
SCT=SCE+SCR
$$

Avec : 

$$
SCE=\sum^{n}_{i=1}(\hat{y}_{i}-\bar{y}_{i})^{2}\\
SCR=\sum^{n}_{i=1}(y_{i}-\hat{y}_{i})^{2}$$

#### Coefficient de détermination : 

$$
R^{2}=\frac{SCE}{SCT}=\frac{\sum^{n}_{i=1}(\hat{y}_{i}-\bar{y}_{i})^{2}}{\sum^{n}_{i=1}(y_{i}-\hat{y}_{i})^{2}}
$$

Propriétés : 

- $R^{2}\in [\ 0 \ ; \ 1\ ]$
- $R^{2}=1 \ \Rightarrow$ liaison parfaite (les observations forment une droite).      
- $R^{2}=0 \ \Rightarrow$ aucune liaison linéaire.
- On démontre que $R^{2}$ est égal à la pente des moindres carrés.

### Propriétés des écarts résiduels :

Soit l'**écart résiduel** : $e_{i}=y_{i}-\hat{y}_{i}=y_{i}-[\bar{y}-b(x_{i}-\bar{x})]$

$$
\sum_{i=1}^{n}e_{i}=\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})=\sum_{i=1}^{n}y_{i}-n\bar{y}-b\sum_{i=1}^{n}(x_{i}-\bar{x})=n\bar{y} - n\bar{y}=0
$$

Propriétés : 

- $E[e_{i}]=0$       
- La **variance empirique** des $e_{i}$ avec : 

$$S^{2}_{Y|X}=\frac{1}{n}\sum^{n}_{i=1}e_{i}^{2}=\frac{1}{n}\left[ \sum^{n}_{i=1}(y_{i}-\bar{y}_{i})^{2}-2b\sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})+b^{2}\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}\right] \\
=S^{2}_{Y}-2bS_{XY}b^{2}+S^{2}_{X}
$$

**Rappel :** 

- $b=r\frac{S_{Y}}{S_{X}}$       
- $r=\frac{S_{XY}}{S_{X}S_{Y}}$      

$$\Rightarrow \ S^{2}_{Y|X}=S^{2}_{Y}-2r\frac{S_{Y}}{S_{X}}rS_{X}S_{Y}+r^{2}S^{2}_{Y}=(1-r^{2})S^{2}_{Y}$$

### Coefficient de corrélation linéaire : 

$$\rho=\frac{Cov(X,Y)}{\sigma_{X}\sigma_{Y}}$$

Propriétés : 

- $\rho \in [-1 \ ; \ 1]$       
- Donne une indication sur le sens et l'intensité de la relation.       
- Généralement, $\rho$ mesure la qualité de l'approximation par une fonction linéaire.

### Dépendance entre deux variables : 

On sait que :

- $r=0 \ \nRightarrow\ Y \perp X$       
- $Y \perp X \ \Rightarrow\ r=0$       
- $Y \perp X \ \Leftrightarrow\ P(Y=y)=P(Y=y \ | \ X=x)$       

Pour être moins restrictifs, on travail avec $E[\ Y|X\ ]$ afin de savoir si $Y$ dépend **en moyenne** de $X$.

#### Rapport de corrélation : 

- si $Y$ dépend en moyenne de $X$, alors il existe un **lien fonctionnel** du type : $Y=f(X)$.      
- On démontre que : $E[E[\ Y|X\ ]]=Y$       
- On démontre que : $V[\ Y|X\ ]=V[E[\ Y|X\ ]]+E[V[\ Y|X\ ]]+$       

On déduit de ces observations le **rapport de corrélation** :

$$\eta^{2}_{Y|X}=\frac{V[E[\ Y|X \ ]]}{V[\ Y|X \ ]}$$

Propriétés : 

- $\eta \in [0 \ ; \ 1]$       
- $\eta = 0 \ \Rightarrow $ indépendance en moyenne ; $E[\ Y|X \ ]$ est *certainement* constante.
- $\eta = 1 \ \Rightarrow E[V[\ Y|X \ ]]$ et $E[\ Y|X \ ]$ sont *certainement* nulles.         
- $E[\ Y|X \ ]=\alpha+X\beta$ (ex. $(X \ ; \Y)$ Gaussiens)$ \ \Rightarrow \ \eta^{2}_{Y|X}=\rho^{2}_{Y|X}$

### Modèle théorique :

Soit $X$ et $Y$, formant des couples d'observations tels que : $(X \ ; \ Y)=(X_{1} \ ; \ Y_{1}),...,(X_{n} \ ; \ Y_{n})$

On cherche $f$ telle que : $f(X) \approx Y \ \Leftrightarrow \ E[\ (Y-f(X))^{2}  \ ] \  \rightarrow 0$

On démontre que $E[ \ (Y − f(X))^{2}]$ est minimale pour $f(X)=E[ \ Y |X \ ]$.

Soit la **fonction de régression** $f$ : 

$$f : X \rightarrow E[ \ Y |X=x \ ]$$

Alors : 

$$Y = E[ \ Y |X \ ] + \epsilon$$

Avec $\epsilon$, le résidu aléatoire tel que : 

- $E[\ \epsilon \ ]=0$ (conséquence de $E(Y ) = E [E(Y |X)]$)      
- $\epsilon \perp \{X \ ; \ E [Y |X] \}$       
- $V[\ \epsilon\ ]=V[\ Y-E[\ Y|X\ ] \ ]=(1-\eta^{2}_{Y|X})V[\ Y\ ]$
- $\epsilon \sim \mathcal{N}(0 \ ; \ \sigma^{2}) \quad \Leftrightarrow \quad Y_{i} \sim \mathcal{N}(\alpha+\beta x_{i} \ ; \ \sigma^{2})$ 

#### Régression linéaire : 

$$E[\ Y|X \ ]=\alpha+X\beta \ \ \Longrightarrow \ \ Y= \alpha+X\beta + \epsilon$$

Egalement : 

$$E[\ Y \ ] = E [E(Y |X)] \ \ \Longrightarrow \ \ E[\ Y \ ] = E[\ α + βX \ ] = α + βE[\ X \ ]$$

Donc la droite passe par le point $(\bar{X} \ ; \ \bar{Y})$. 

On a également : 

$$Y − E[ \ Y \ ] = α + βX + ϵ − [ \ α + βE(X \ ]] = β [ \ X − E(X) \ ] + \epsilon$$

$$E[\ (Y-E[\ Y\ ])(X-E[\ X\ ])\ ]=\beta E[\ (X-E[\ X\ ])^{3}\ ]+E[\ \epsilon(X-E[\ X\ ])\ ]$$

Sachant que : 

- $V[\ X \ ]=E[\ (X - E [ \ X\ ])^{2}\ ]$      
- $Cov(X \ ; \ Y)=E[\ (Y-E[\ Y\ ])(X-E[\ X\ ])\ ]$      
- $Cov(\epsilon \ ; \ X)=E[\ (\epsilon -E[\ \epsilon\ ])(X-E[\ X\ ])\ ]$      
- $E[\ \epsilon \ ]=0$

$$Cov(X \ ; \ Y)=\beta \ V[\ X\ ] + Cov(\epsilon \ ; \ X)$$

Alors même que $\epsilon$ n'est pas corrélée à $X$, soit $Cov(\epsilon \ ; \ X)=0$ et donc : 

$$Cov(X \ ; \ Y)=\beta \ V[\ X\ ]$$

$$\beta= \frac{Cov(X \ ; \ Y)}{ V[\ X\ ]}$$

Egalement : $\rho=\frac{Cov(X \ ; \ Y)}{\sigma_{X}\sigma_{Y}} \ \Rightarrow \ Cov(X \ ; \ Y) = \rho \sigma_{X}\sigma_{Y}$

$$\Rightarrow \ \ \beta=\frac{Cov(X \ ; \ Y)}{\sigma^{2}_{X}}=\frac{\rho \sigma_{X}\sigma_{Y}}{\sigma^{2}_{X}} \ \Rightarrow \ \beta=\rho\frac{\sigma_{Y}}{\sigma_{X}}$$

**Rappel :** L'équation de la droite de régression est donnée par :

$$Y-E[\ Y \ ]= \rho \frac{\sigma_{Y}}{\sigma_{X}}\left(X-E[\ X \ ]\right)+\epsilon$$

$$\Rightarrow \ Y= \left(E[\ Y \ ] - \rho \frac{\sigma_{Y}}{\sigma_{X}}E[\ X \ ]\right)+ \rho \frac{\sigma_{Y}}{\sigma_{X}}X + \epsilon$$

Si la régression est linéaire, on retrouve le résultat précédent ($\eta^{2}_{Y|X}=\rho^{2}$) :

$$
V[\ Y\ ]=V\left[\ E[\ Y \ ]+ \rho \frac{\sigma_{Y}}{\sigma_{X}}\left(X-E[\ X \ ]\right)+\epsilon\ \right]
$$

Avec $X$ et $\epsilon$ non corrélées :

$$
V[\ Y\ ]=V\left[\ \rho \frac{\sigma_{Y}}{\sigma_{X}}\left(X-E[\ X \ ]\right)+\epsilon\ \right] \\
=V\left[\ \rho \frac{\sigma_{Y}}{\sigma_{X}}\left(X-E[\ X \ ]\right) \right]+V\left[\ \epsilon\ \right]=\rho^{2}\frac{\sigma^{2}_{Y}}{\sigma^{2}_{X}}V\left[\ X \right]-V \left[\ E[\ X \ ]\right] +V\left[\ \epsilon\ \right]=\rho^{2}V\left[\ Y \right]+V\left[\ \epsilon\ \right]\\
V[\ \epsilon\ ]=(1-\rho^{2}) \ V[\ Y\ ]
$$

Or on a vu que : $V[\ \epsilon\ ]=(1-\eta^{2}_{Y|X}) \ V[\ Y\ ]$

Donc : 

$$\rho^{2}=\eta^{2}_{Y|X}$$  

### Modèle linéaire :

*Les résultats obtenus jusqu'ici s'appliquent même si $X$ n'est pas aléatoire mais contrôlée par l'expérimentateur*.

$$
\forall i : Y_{i}=\alpha + \beta x_{i} +\epsilon_{i}
$$

On parle de **modèle linéaire** et non de régression linéaire. 

Propriétés de la variable aléatoire $\epsilon$ :

- $\epsilon \sim \mathcal{N}(0 \ ; \ \sigma^{2})$
- $\forall i : V[\ \epsilon \ ]=\sigma^{2}$ (homoscédasticité)         
- $\epsilon_{i} \perp \epsilon_{j} \ , \forall i \neq j \ \Rightarrow \ Cov(\epsilon_{i},\epsilon_{j})=0$

### Aspects inférentiels de la régression linéaire :

En partant du modèle :

$$Y_{i}=\alpha + \beta x_{i} +\epsilon_{i}$$

On effectue un développement par la méthode des moindres carrés, comparable à celui déja effectué, afin de calculer les paramètres en tenant compte du caractère aléatoire de $Y$ :

$$B=\frac{\sum^{n}_{i=1} (Y_{i}-\bar{Y})(x_{i}-\bar{x})}{\sum^{n}_{i=1}(x_{i}-\bar{x})^{2}}$$

$$A=\bar{Y}-B\bar{x}$$

#### Propriétés de $A$ et $B$ :

- **Linéarité** car ils sont combinaisons linéaires des $Y_{i}$ :

$$\delta_{i}=\sum^{n}_{i=1}\frac{(x_{i}-\bar{x})}{\sum^{n}_{i=1}(x_{i}-\bar{x})^{2}}Y_{i.} \ \qquad \Longrightarrow \qquad B=\Sigma^{n}_{i=1}\delta_{i}Y_{i} \qquad et \qquad A=\Sigma^{n}_{i=1}\left(\frac{1}{n}\bar{x}\delta_{i}\right)Y_{i}$$

- Absence de **biais** : 

$$E[\ B \ ]=E\left[\  \frac{\sum^{n}_{i=1} (Y_{i}-\bar{Y})(x_{i}-\bar{x})}{\sum^{n}_{i=1}(x_{i}-\bar{x})^{2}} \ \right]=  \frac{\sum^{n}_{i=1} E\left[\ Y_{i}-\bar{Y} \ \right](x_{i}-\bar{x})}{\sum^{n}_{i=1}(x_{i}-\bar{x})^{2}}$$

On sait que :

$$E[\ Y_{i} \ ]=E[\ \alpha+\beta x_{i}+\epsilon_{i} \ ]=\alpha+\beta x_{i}$$

$$E[\ \bar{Y} \ ]=\frac{1}{n}\Sigma_{i}E[\ Y_{i} \ ]=\frac{1}{n}\Sigma_{i}(\alpha+\beta x_{i})=\alpha+\beta \bar{x}$$

$$\Rightarrow E[\ B \ ]=  E\left[\ \frac{\sum^{n}_{i=1} \beta (x_{i}-\bar{x})(x_{i}-\bar{x})}{\sum^{n}_{i=1}(x_{i}-\bar{x})^{2}} \ \right]=\beta$$

$$
E[\ A\ ]=E[\ \bar{Y}-\beta\bar{x}\ ]=E[\ Y\ ]-\beta x=(\alpha+\beta x+E[\ \epsilon\ ])-\beta x=\alpha
$$

On démontre que : 

$$
V[\ B\ ]=\frac{\sigma^{2}}{nS^{2}_{X}}
$$

$$
V[\ A\ ]=\frac{\sigma^{2}}{n}\left(1+\frac{\bar{x}}{S^{2}_{X}}\right)
$$

Où $\sigma^{2}$ est la variance des résidus : $S^{2}_{X}=\frac{1}{n}\Sigma_{i}(x_{i}-\bar{x})^{2}$

- **non indépendance** : $A \not\perp B$

On démontre que : 

$$Cov(A \ ; \ B)=\frac{-\bar{x}\sigma^{2}}{\Sigma^{n}_{i=1}(x_{i}-\bar{x})^{2}}$$

$$\rho(A \ ; \ B)=\frac{-\bar{x}}{\sqrt{\Sigma^{n}_{i=1}(x_{i}^{2})/n}}$$

- $A$ et $B$ sont des estimateurs de **variance minimale** (théorème de *Gauss-Markov*).

### Estimation par maximum de vraissemblance : 

**Rappel :**

- $\epsilon \sim \mathcal{N}(0 \ ; \ \sigma^{2})$

```{r }
modele=lm(Prix~Superficie, data = df)
```

Affichage des paramètres du modèle : 

```{r }
summary(modele)
anova(modele)
predict(modele,interval="confidence",level=0.95)
```

Commentaires :

Significativité du coefficient de régression linéaire : 
P-Value = 7.86e-13


Intervalles de confiance et de prédiction :

```{r }
confint(modele)

modele_df = as.data.frame(cbind(
  df,
  predict(modele, interval = "confidence", level = 0.95)
))
```

Représentation graphique du modèle :

```{r }
ggplot(df, aes(x=Superficie, y=Prix))+ 
  geom_point()+
  geom_smooth(method=lm, se=T)
```

```{r }
PI = as.data.frame(cbind(Prix = df$Prix, Superficie = df$Superficie,
predict(modele,interval="prediction")))

ggplot(PI, aes(x=Superficie, y=Prix))+
geom_line(aes(y=lwr), color = "red", linetype = "dashed")+
geom_line(aes(y=upr), color = "red", linetype = "dashed")+
geom_point()+geom_smooth(method=lm, se=T) + geom_text(label = row.names(df),
vjust = - 1, check_overlap = TRUE, size = 3)
```

Etude des résidus :

```{r }
library(ggfortify)

autoplot(
  modele,
  which = 1,
  ncol = 1,
  label.size = 3,
  label.hjust = -0.8,
  label.n = 6
)

autoplot(
  modele,
  which = 2,
  ncol = 1,
  label.size = 3,
  label.hjust = -0.8,
  label.n = 6
)
```

```{r }
install.packages("olsrr")

library(DescTools)
library(lmtest)
library(olsrr)

# Recherhe d'autocorrélation :
DurbinWatsonTest(modele, alternative = 'two.sided')

# Evaluation de l'homogeneite : 
bptest(modele)

# Test de la normalite :
shapiro.test(modele$residuals)

```
Recherche de valeurs atypiques :

```{r }
autoplot(
  modele,
  which = 3,
  ncol = 1,
  label.size = 3,
  label.hjust = -0.8,
  label.n = 6
) +
  aes(.fitted, .stdresid) + labs(x = "fitted values", y = "standardized residuals") +
  ggtitle("")

# Graphique basique des leviers :
plot(hatvalues(modele))

# Version amélioree du graphique des leviers :
ggplot(modele, aes(seq_along(.hat), .hat)) + geom_col(width = 0.1, colour = "blue") +
  labs(x = "Observation", y = "Leverage") + geom_text(
    label = rownames(df),
    check_overlap = T,
    vjust = -0.8,
    size = 3
  ) + geom_hline(yintercept = 4 / nrow(df),
                 colour = "red")

# Graphique pour la distznce de cook :  
ols_plot_cooksd_chart(modele)
 

ols_plot_dfbetas(modele)
ols_plot_dffits(modele)
ols_plot_resid_lev(modele)
```


