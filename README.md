# Sequential_Recommendation_Childcare



A research pipeline over the **American Time Use Survey (ATUS) 2003–2024** that answers a
behavioural question: *given a parent's 24-hour activity diary, what concrete change would
increase the time they spend on childcare?*

The output is not a score or a label but a counterfactual edit to the person's actual day —
for example *"after Eating/Drinking, do Childcare instead of Housework — specifically,
Playing with HH children (ATUS code 030103)"* — grounded in what behaviourally similar
parents actually did.

The scientific contribution is the **comparison of four ways to represent a day** (B1–B4).
Clustering, exemplar selection, recommendation and evaluation are held identical across all
four, so the representation is the only variable.

- **Population:** parents only (`TRCHILDNUM > 0`) — 107,584 respondents
- **Target behaviour:** Childcare (HH), category 4 of a 19-category scheme
- **Split:** 70/15/15 by respondent, seeded — 16,139 respondents in the test split
