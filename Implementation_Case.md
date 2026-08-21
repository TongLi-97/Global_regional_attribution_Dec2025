# Implementation Case: Attribution of Responses to GHG and OANT Forcing over N.W. North America Using the Global and Regional Constraint Scheme

As an illustrative example, we conduct the regional attribution of GHG and OANT forcings in the AR6 region of N.W. North America (NWN), using both global and regional constraints (see the schematic shown in Fig. S3).

## Observational vector

The observational vector $Y_o \in \mathbb{R}^{197}$ is constructed by concatenating the annual mean global land surface temperature time series (140 years, from 1885 to 2025 when applying the 30% coverage threshold to calculate the global land mean; Fig. S1b) with the NWN regional annual mean series for years available (57 years, starting discontinuously from 1959; Extended Data Fig. 1b). This results in a multi-region observational vector of length 197.

## Prior mean vector

The prior mean vector $\mu_{ALL,TF} \in \mathbb{R}^{943}$ consists of two components. The first block corresponds to the GHG, OA, and NAT forcing responses of global land and NWN, with each forcing contributing a vector of length 197 that corresponds to the observations, for a total of 591 elements. Their sum represents the ALL response that is compared with observations. The second block corresponds to the target region (NWN) and contains the full time series (1850–2025; 176 years) of GHG and OA responses, concatenated to length 352.

## Observation operator

The observation operator

$$
H \in \mathbb{R}^{197 \times 943} = \begin{bmatrix} I & I & I & 0 \end{bmatrix}
$$

is constructed to project the prior mean vector into the observational space. Specifically, three identity matrices $I \in \mathbb{R}^{197 \times 197}$ select the GHG, OA, and NAT components corresponding to the global land and NWN observational time series (total: $3 \times 197 = 591$ columns). A zero matrix $0 \in \mathbb{R}^{197 \times 352}$ ensures that the target-region components (time series for 1850–2025 in NWN) are set to zero.

This structure ensures that the prior is matched against the observed series only through the global plus regional forcing components, while the target-region time series in the prior do not directly contribute to the observation equation. The additive forcing assumption means the observations see the sum of GHG, OANT, and NAT responses from the global land plus NWN regions.

## Observational error covariance

$\Sigma_o \in \mathbb{R}^{197 \times 197}$ represents the uncertainty in the observed surface temperature time series of the global land and NWN regions. This uncertainty accounts for both internal climate variability and observational measurement error. Internal variability is estimated using 320 simulations from six large-ensemble climate model experiments, while measurement uncertainty is characterized using 200 realizations from the HadCRUT5 observational ensemble. Concatenation of global and regional series allows $\Sigma_o$ to retain cross-regional covariance structure.

## Prior covariance matrix

$\Sigma_{m,ALL,TF} \in \mathbb{R}^{943 \times 943}$ characterizes the joint uncertainty across all model-simulated responses in the prior vector $\mu_{ALL,TF}$. The diagonal blocks capture within-component uncertainties and correlations among different forcings (GHG, OANT, NAT) at each time point. The off-diagonal blocks preserve critical cross-covariances between observational-period responses and target time series, enabling observational constraints to propagate information across different time periods, spatial scales, and forcing combinations through the Bayesian update.

## Posterior distribution

After applying the Bayesian update and selecting the target-forcing components, the posterior mean and covariance are given by:

$$
\widehat{\mu}_{TF \mid Y_o}
= G\left[
\widehat{\mu}_{ALL,TF}
+ \widehat{\Sigma}_{m,ALL,TF}H^T
\left(H\widehat{\Sigma}_{m,ALL,TF}H^T + \widehat{\Sigma}_o\right)^{-1}
\left(Y_o-H\widehat{\mu}_{ALL,TF}\right)
\right].
\tag{4}
$$

$$
\widehat{\Sigma}_{m,TF \mid Y_o}
= G\left[
\widehat{\Sigma}_{m,ALL,TF}
- \widehat{\Sigma}_{m,ALL,TF}H^T
\left(H\widehat{\Sigma}_{m,ALL,TF}H^T + \widehat{\Sigma}_o\right)^{-1}
H\widehat{\Sigma}_{m,ALL,TF}
\right]G^T.
\tag{5}
$$

Here, $TF$ denotes the target forcing (GHG or OANT). The matrix

$$
G \in \mathbb{R}^{197 \times 943}
= \begin{bmatrix} 0 & 0 & 0 & I \end{bmatrix},
\qquad I \in \mathbb{R}^{197 \times 352},
$$

acts as a selection operator that extracts the target-region target-forcing block from the joint state vector. Equations (4)–(5) yield the posterior distribution of the GHG- and OANT-induced warming responses in the target region, northwestern North America (NWN), over the period 1850–2025. The resulting posterior mean, $\widehat{\mu}_{TF \mid Y_o}$, represents the best estimate of the attributable warming, while $\widehat{\Sigma}_{m,TF \mid Y_o}$ quantifies the associated uncertainty after incorporating observational constraints.
