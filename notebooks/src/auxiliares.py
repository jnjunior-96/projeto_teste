
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import (
    shapiro,
    levene,
    norm,
    mannwhitneyu,
    ttest_ind,
    ttest_rel,
    f_oneway,
    wilcoxon,
    friedmanchisquare,
    kruskal
)
from collections import namedtuple


def histograma_boxplot(dataframe, coluna, intervalos='auto'):
    fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    gridspec_kw={
        "height_ratios": (0.15, 0.85),
        "hspace": 0.02
    }
    )

    sns.boxplot(
        data=dataframe,
        x=coluna,
        showmeans=True,
        meanline=True,
        meanprops={"color": "C1", "linewidth": 1.5, "linestyle": "--"},
        medianprops={"color": "C2", "linewidth": 1.5, "linestyle": "--"},
        ax=ax1,
    )

    sns.histplot(data=dataframe, x=coluna, kde=True, bins=intervalos, ax=ax2)

    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", color="gray", alpha=0.3)
        ax.set_axisbelow(True)

    ax2.axvline(dataframe[coluna].mean(), color="C1", linestyle="--", label="Média")
    ax2.axvline(dataframe[coluna].median(), color="C2", linestyle="--", label="Mediana")
    ax2.axvline(dataframe[coluna].mode()[0], color="C3", linestyle="--", label="Moda")

    ax2.legend()

    plt.show()


def analise_shapiro(dataframe, alfa=0.05):
    print('Teste de Shapiro-Wilk:')

    for coluna in dataframe.columns:
        estatistica, valor_p_sw =shapiro(dataframe[coluna], nan_policy='omit')
        print(f'{estatistica:.3f}')
        if valor_p_sw > alfa:
            print(f'{coluna} segue uma distribuição normal (valor p : {valor_p_sw:.3f})')
        else:
            print(f'{coluna} Não segue uma distribuição normal (valor p : {valor_p_sw:.3f})')


def analise_levene(dataframe, alfa=0.05, center='mean'):
    print('Teste de Levene')

    estatistica, valor_p_levene =levene(*[dataframe[coluna] for coluna in dataframe.columns], center=center ,nan_policy='omit')

    print(f'{estatistica:.3f}')
    if valor_p_levene > alfa:
        print(f'Variancias iguais (valor p : {valor_p_levene:.3f})')
    else:
        print(f'Ao menos uma varianca é diferente (valor p : {valor_p_levene:.3f})')


def analises_shapiro_levene(dataframe, alfa=0.05, centro='mean'):
    analise_shapiro(dataframe, alfa)

    print()

    analise_levene(dataframe, alfa, centro)


def teste_z(dados, media_pop, desvio_pop):
    n = len(dados)
    media_amostral = np.mean(dados)
    desvio_media_amostral = desvio_pop / np.sqrt(n)
    z = (media_amostral - media_pop) / desvio_media_amostral
    valor_p = 1 - norm.cdf(z)
    TesteZ = namedtuple('TesteZ',['estatistica','valor_p'])
    return TesteZ(z, valor_p)


def analise_ttest_ind(dataframe, alfa=0.05, variancias_iguais=True, alternativa='two-sided'):

    print('Teste t de Student')


    estatistica_ttest, valor_p_ttest = ttest_ind(
        *[dataframe[coluna] for coluna in dataframe.columns],
        equal_var=variancias_iguais,
        alternative=alternativa,
        nan_policy='omit')

    print(f'{estatistica_ttest=:.3f}')
    if valor_p_ttest > alfa:
        print(f'Não rejeita a hipótese nula (valor p : {valor_p_ttest:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p : {valor_p_ttest:.3f})')


def analise_ttest_rel(dataframe, alfa=0.05, alternativa='two-sided'):
    print('Teste t de Student')

    estatistica_ttest, valor_p_ttest = ttest_rel(
        *[dataframe[coluna] for coluna in dataframe.columns],
        alternative=alternativa,
        nan_policy='omit')

    print(f'{estatistica_ttest=:.3f}')
    if valor_p_ttest > alfa:
        print(f'Não rejeita a hipótese nula (valor p : {valor_p_ttest:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p : {valor_p_ttest:.3f})')


def analise_anova_one_way(dataframe, alfa=0.05):
    print('Teste ANOVA one way')

    estatistica_f, valor_p_f = f_oneway(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy='omit')

    print(f'{estatistica_f=:.3f}')
    if valor_p_f > alfa:
        print(f'Não rejeita a hipótese nula (valor p : {valor_p_f:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p : {valor_p_f:.3f})')


def analise_wilcoxon(dataframe, alfa=0.05, alternativa='two-sided'):
    print('Teste Wilcoxon')

    estatistica_wilcoxon, valor_p_wilcoxon = wilcoxon(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy='omit', alternative=alternativa)

    print(f'{estatistica_wilcoxon=:.3f}')
    if valor_p_wilcoxon > alfa:
        print(f'Não rejeita a hipótese nula (valor p : {valor_p_wilcoxon:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p : {valor_p_wilcoxon:.3f})')


def analise_mannwhitneyu(dataframe, alfa=0.05, alternativa='two-sided'):
    print('Teste Mannwhitneyu')

    estatistica_mannwhitneyu, valor_p_mannwhitneyu = mannwhitneyu(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy='omit', alternative=alternativa)

    print(f'{estatistica_mannwhitneyu=:.3f}')
    if valor_p_mannwhitneyu > alfa:
        print(f'Não rejeita a hipótese nula (valor p : {valor_p_mannwhitneyu:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p : {valor_p_mannwhitneyu:.3f})')


def analise_friedman(dataframe, alfa=0.05):
    print('Teste Friedman')

    estatistica_fd, valor_p_fd = friedmanchisquare(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy='omit')

    print(f'{estatistica_fd=:.3f}')
    if valor_p_fd > alfa:
        print(f'Não rejeita a hipótese nula (valor p : {valor_p_fd:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p : {valor_p_fd:.3f})')


def analise_kruskal(dataframe, alfa=0.05):
    print('Teste Kruskal')

    estatistica_kk, valor_p_kk = kruskal(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy='omit')

    print(f'{estatistica_kk=:.3f}')
    if valor_p_kk > alfa:
        print(f'Não rejeita a hipótese nula (valor p : {valor_p_kk:.3f})')
    else:
        print(f'Rejeita a hipótese nula (valor p : {valor_p_kk:.3f})')