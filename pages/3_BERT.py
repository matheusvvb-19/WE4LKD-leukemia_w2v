# IMPORTS:
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# GLOBAL VARIABLES:

# FUNCTIONS:
@st.cache()
def get_sentences_dataset():
    data = [
        [1970, "Combination chemotherapy using L-asparaginase, daunorubicin, and cytarabine in adults with AML."],
        [1972, "Fourty-four patients with AML were treated with RP 22050, a new, semisynthetic derivative of daunorubicin."],
        [1974, "The use of a combination of daunorubicin and cytarabine in the treatment of AML in a district general hospital is described."],
        [1975, "Thirty-seven adults with AML (AML) were treated with a combination of daunorubicin, cytarabine."],
        [1976, "Maintenance of remission in adult AML using intermittent courses of cytarabine (cytarabine) and thioguanine (thioguanine)."],
        [1978, "Normal infant after treatment of AML in pregnancy with daunorubicin."],
        [1979, "Enhanced cytarabine accumulation after administration of MTX was also observed in human AML cells."],
        [1980, "Cyclophosphamide, cytarabine and methotrexate versus cytarabine and thioguanine for AML in adults."],
        [1981, "Sequential combination of pyrazofurin and azacitidine in patients with AML and carcinoma."],
        [1981, "Three patients with AML received high doses of daunorubicin, first in the free form and later as complex with DNA."],
        [1981, "vincristine, and 6alpha-methylprednisone (ROAP) to 91 patients with AML who were 50 yr of age or older."],
        [1982, "Relapses in nine patients with AML were treated with a combination of aclarubicin (ACR) and cytarabine (cytarabine)."],
        [1982, "Treatment of relapsed AML with a combination of aclarubicin and cytarabine."],
        [1983, "The effects of dexamethasone and tetradecanoyl phorbol acetate on plasminogen activator release by human AML cells."],
        [1983, "Therapeutic responses included a partial remission in a patient with AML (AML) refractory to cytarabine."],
        [1983, "dexamethasone, fluorouracil or methotrexate on human AML cell line KG-1."],
        [1984, "Delayed growth arrest was observed in HL-60 AML cells after exposure to thioguanine (thioguanine)."],
        [1984, "The current status of, and the future prospects for, low-dose cytarabine therapy in patients with AML is discussed."],
        [1984, "The use of intermediate dose cytarabine (ID cytarabine) in the treatment of AML in relapse."],
        [1985, "Low-dose cytarabine regimen induced a complete remission with normal karyotypes in a case with hypoplastic AML with No."],
        [1985, "Preliminary results in secondary AML (ANLL) treated with Idarubicin."],
        [1985, "The utility and indication for low-dose cytarabine therapy of AML remains to be determined."],
        [1985, "Treatment of AML with a daunorubicin cytarabine thioguanine regimen without maintenance therapy."],
        [1985, "doxorubicin and cytarabine contribute equally to prediction of response in AML with improved confidence level."],
        [1986, "A case of persistent cerebellar toxicity following systemic high dose cytarabine (HDCA) treatment of AML is reported."],
        [1986, "A comparison of in vitro sensitivity of AML precursors to mitoxantrone, 4'deoxydoxorubicin, idarubicin and daunorubicin."],
        [1986, "A critical appraisal of low-dose cytarabine in patients with AML and myelodysplastic syndromes."],
        [1986, "A patient with myelofibrosis transforming into AML (ANLL) was treated with low doses of cytarabine (cytarabine)."],
        [1986, "Patients with AML received cytarabine, either 2 or 6 g/m2/72 h by continuous infusion."],
        [1987, "Etoposide in combination with cytarabine, doxorubicin, and thioguanine for treatment of AML in a protocol adjusted for age."],
        [1987, "Fifteen patients with AML were given a total of 17 courses of high-dose cytarabine (cytarabine)."],
        [1987, "Four patients with AML (AML) and three with myelodysplastic syndrome (MDS) were given low dose cytarabine (cytarabine) therapy."],
        [1987, "Low dose arabinosyl cytosine (cytarabine) is effective for treatment of AML (ANLL)."],
        [1987, "Low-dose cytarabine (LD-Ara C) treatment in dysmyelopoietic syndromes (DMPS) and AML (AML)."],
        [1987, "Thirty-eight patients with AML (AML) were treated with mitoxantrone (Mto) combined with cytarabine (cytarabine)."],
        [1988, "A case of persistent cerebellar dysfunction following high-dose cytarabine (cytarabine) treatment of AML is reported."],
        [1988, "Daunorubicin in patients with relapsed and refractory AML previously treated with anthracycline."],
        [1988, "In vivo uptake of daunorubicin by AML (AML) cells measured by flow cytometry."],
        [1988, "Interleukin 3 enhances the cytotoxic activity of cytarabine (cytarabine) on AML (AML) cells."],
        [1988, "cytarabine is the most important drug in the clinical chemotherapy of AML."],
        [1989, "23 adult patients with refractory or relapsed AML (AML) received salvage chemotherapy with mitoxantrone and etoposide."],
        [1989, "Eight patients with AML were treated with a combination of daunorubicin (1.5 mg/kg body weight) and cytarabine."],
        [1989, "High-dose cytarabine (HDARA-C) is an effective but toxic treatment for AML (AML)."],
        [1989, "Low-dose cytarabine therapy for AML in elderly patients."],
        [1989, "Standard induction and low dose cytarabine treatment in patients over 60 with AML or MDS."],
        [1989, "The urine of a patient who suffered from AML was red coloured after administration of mitoxantrone and etoposid."],
        [1990, "Concordant changes of pyrimidine metabolism in blasts of two cases of AML after repeated treatment with cytarabine in vivo."],
        [1990, "Cytotoxicity and DNA damage caused by idarubicin and its metabolite 4-demethoxy-13-hydroxydaunorubicin in human AML cells."],
        [1990, "Rapid Remission Induction and Improved Disease Free Survival in AML Using Daunorubicin, cytarabine, and lomustine."],
        [1990, "Schedule-dependent interaction of cytarabine plus doxorubicin or cytarabine plus mitoxantrone in AML cells in culture."],
        [1990, "The original LBN AML is sensitive to the chemotherapeutic agent cyclophosphamide (CY)."],
        [1990, "Treatment of poor-prognosis, newly diagnosed AML with high-dose cytarabine (cytarabine) and rHUGM-CSF."],
        [1991, "14 patients with AML (AML) and 7 with myelodysplastic syndrome (MDS) were treated with cytarabine in low dosage."],
        [1991, "Growth factors also affect the sensitivity of AML blast cells to cytarabine (cytarabine)."],
        [1991, "High-dose cytarabine and daunorubicin induction and postremission chemotherapy for the treatment of AML in adults."],
        [1991, "Low dose cytarabine (LDARAC) has been commonly used in the treatment of AML (AML) in elderly patients."],
        [1991, "Oral idarubicin and low dose cytarabine as the initial treatment of AML in elderly patients."],
        [1991, "The pharmacokinetics of thioguanine were studied in 10 patients with AML treated with 25-100 mg/m(2) orally."],
        [1991, "Treatment with HGFs resulted in [3H] cytarabine DNA incorporation that was significantly higher in AML blasts versus NBMMC."],
        [1991, "We report a patient with AML who had typical AE due to cytarabine with histopathologic ESS."],
        [1991, "hydrocortisonecypionate in culture protects the blast cells in AML from the lethal effects of cytarabine."],
        [1992, "A six year old Chinese boy with relapsed AML (APML) failed to respond to reinduction with Daunorubicin and Cytarabine infusion."],
        [1992, "Bone marrow transplantation versus high-dose cytarabine-based consolidation chemotherapy for AML in first remission."],
        [1992, "Both MAbs reacted positively in 1 patient with AML (ANLL) at diagnosis who achieved remission with teniposide and cytarabine."],
        [1992, "Long-term outcome of high-dose cytarabine-based consolidation chemotherapy for adults with AML."],
        [1992, "Maintenance with low-dose cytarabine for AML in complete remission."],
        [1992, "Thus, AML blasts respond to fazarabine in culture with a pattern similar to that of 5-aza and opposite to that of cytarabine."],
        [1992, "Treatment of AML in the elderly with low-dose cytarabine, hydroxyurea, and calcitriol."],
        [1992, "We conclude that intermediate dose cytarabine did not substantially improve results of induction chemotherapy for AML."],
        [1992, "and daunorubicin-mediated cytotoxicity of AML cells and normal myeloid progenitors."],
        [1993, "A cell culture model for the treatment of AML with fludarabine and cytarabine."],
        [1993, "Fludarabine potentiates metabolism of cytarabine in patients with AML during therapy."],
        [1993, "Infusion of fludarabine before cytarabine augments the rate of ara-CTP synthesis in circulating AML blasts during therapy."],
        [1993, "No difference of intracellular daunorubicin accumulation was observed between CD34+ and CD34- AML cells of 4 P-170- patients."],
        [1993, "One of the differences between AML (AML) and acute lymphoblastic leukemia (ALL) is their sensitivity to vincristine."],
        [1993, "The optimal dose and schedule of cytarabine in induction chemotherapy of newly diagnosed AML is not established."],
        [1993, "The regimen containing BHAC (or cytarabine), DNR, and 6-MP may be useful as induction chemotherapy for AML-M0."],
        [1993, "Therapy of refractory/relapsed AML and blast crisis of chronic myeloid leukemia with the combination of cytarabine."],
        [1993, "This led us to administer fludarabine and cytarabine to 59 patients with AML in relapse or unresponsive to initial therapy."],
        [1993, "cytarabine (cytarabine) and etoposide are often used in combination in the treatment of AML (AML)."],
        [1994, "A patient is described with undifferentiated AML refractory to two courses of daunorubicin and cytarabine."],
        [1994, "Benefit of high-dose cytarabine-based consolidation chemotherapy for adults with AML."],
        [1994, "Combined antileukemic activity of pIXY 321 and cytarabine against human AML cells."],
        [1994, "Cytarabine (cytarabine) is currently used in the treatment of adult AML (AML)."],
        [1994, "Detection of cytarabine resistance in patients with AML using flow cytometry."],
        [1994, "In three AML samples at diagnosis and at their relapse, cytarabine incorporation into the DNA was determined."],
        [1994, "Long-term results following treatment of newly-diagnosed AML with continuous-infusion high-dose cytarabine."],
        [1994, "Structural analysis of the deoxycytidine kinase gene in patients with AML and resistance to cytarabine."],
        [1995, "An effective salvage regimen with aclarubicin for daunorubicin-resistant AML in children."],
        [1995, "Clonogenic data from fresh AML LDMCs not pretreated with growth factors demonstrated a heterogenous response to cytarabine."],
        [1995, "Idarubicin in combination with cytarabine and etoposide in the treatment of post myelodysplastic syndrome AML (MDS-AML)."],
        [1995, "In summary, high-dose cytarabine/DNR consolidation can improve the long-term outcome of a subgroup of de novo AML patients."],
        [1995, "Inhibition of bcl-2 with antisense oligonucleotides induces apoptosis and increases the sensitivity of AML blasts to cytarabine."],
        [1995, "Low-dose cytarabine in patients with AML not eligible for standard chemotherapy."],
        [1995, "Successful treatment of retinoic acid syndrome with high-dose dexamethasone pulse therapy in a child with AML treated with ATRA."],
        [1995, "The effect of haemopoietic growth factors on the cell cycle of AML progenitors and their sensitivity to cytarabine in vitro."],
        [1995, "We evaluated the efficacy and toxicity of aclarubicin for AML (ANLL) refractory to daunorubicin in childhood."],
        [1996, "A prospective randomized trial of idarubicin vs daunorubicin in combination chemotherapy for AML of the age group 55 to 75."],
        [1996, "Complete remission was obtained following standard chemotherapy for AML (doxorubicin, cytosin-arabinoside, thioguanine)."],
        [1996, "GM-CSF administered during induction treatment of AML with a DNR/cytarabine combination did not provide any clinical benefit."],
        [1996, "Generation of reactive oxygen intermediates after treatment of blasts of AML with cytarabine."],
        [1996, "Haemopoietic growth factors, the cell cycle of AML progenitors and sensitivity to cytarabine."],
        [1996, "It has been shown recently in China that arsenictrioxide (arsenictrioxide) is a very effective treatment for AML (APL)."],
        [1996, "Peripheral leukaemic cells from 10 patients with AML were incubated with daunorubicin (daunorubicin)."],
        [1996, "Steady progress has been made in the treatment of AML since the discovery of daunorubicin in the 1960s."],
        [1996, "The clinical development of cytarabine for the treatment of AML (AML) provides a useful paradigm for the study of this process."],
        [1996, "cytarabine has been shown to induce apoptosis of human AML HL-60 cells."],
        [1997, "Altered intracellular distribution of daunorubicin in immature AML cells."],
        [1997, "Cytarabine ocfosfate (SPAC) was administered orally to 19 patients with AML (AML) and myelodysplastic syndrome (MDS)."],
        [1997, "Furthermore, 11 AML and four ALL patients were treated with fractionated daunorubicin at a dose of 50 mg/m2/week."],
        [1997, "Long-term outcome of high-dose cytarabine-based consolidation chemotherapy for older patients with AML."],
        [1997, "Molecular remission in PML/RAR alpha-positive AML by combined all-trans retinoic acid and idarubicin (AIDA) therapy."],
        [1997, "Natural resistance of AML cell lines to mitoxantrone is associated with lack of apoptosis."],
        [1997, "The inclusion of HD 6MP and ID cytarabine in the treatment of AML in first remission appears to be feasible."],
        [1998, "Complete remission after treatment of AML with arsenictrioxide."],
        [1998, "Frequency of prolonged remission duration after high-dose cytarabine intensification in AML varies by cytogenetic subtype."],
        [1998, "Inv(16) AML cells show an increased sensitivity to cytarabine in vitro."],
        [1998, "Seven relapsed and/or refractory AML patients were treated by arsenictrioxide (arsenictrioxide)."],
        [1998, "To date, the prognostic impact of cytarabine dose escalation within various cytogenetic groups of AML has not been assessed."],
        [1998, "Treatment of AML in the older patient with attenuated high-dose cytarabine."],
        [1998, "Two reports from China have suggested that arsenictrioxide can induce complete remissions in patients with AML (APL)."],
        [1998, "arsenictrioxide (arsenictrioxide) has recently been shown to induce complete remission in AML (APL)."],
        [1998, "arsenictrioxide as an inducer of apoptosis and loss of PML/RAR alpha protein in AML cells."],
        [1999, "AV block after arsenictrioxide (arsenictrioxide) treatment for refractory AML is very rare."],
        [1999, "Acute pancreatitis in AML (AML) has been rarely associated with cytarabine therapy."],
        [1999, "Anthracyclines such as daunorubicin (daunorubicin) are typically used to treat AML and can induce drug resistance."],
        [1999, "It was recently reported that arsenictrioxide (As(2)O(3)) can induce complete remission in patients with AML (APL)."],
        [1999, "Pancreatitis in the setting of AML therapy may be an infrequent and self-limited toxicity of cytarabine."],
        [1999, "Recently, arsenictrioxide (As) was shown to be an effective treatment of AML (APL)."],
        [1999, "The phosphoinositide 3-kinase/Akt pathway is activated by daunorubicin in human AML cell lines."],
        [1999, "arsenictrioxide has recently been introduced as a promising new agent to treat refractory AML (APL)."],
        [1999, "arsenictrioxide selectively induces AML cell apoptosis via a hydrogen peroxide-dependent pathway."],
        [1999, "dexamethasone does not counteract the response of AML cells to all-trans retinoic acid."],
        [2000, "Growth and endocrine function in children with AML after bone marrow transplantation using busulfan/cyclophosphamide."],
        [2000, "Heterogeneity of isolated mononuclear cells from patients with AML affects cellular accumulation and efflux of daunorubicin."],
        [2000, "P-glycoprotein in primary AML and treatment outcome of idarubicin/cytosine arabinoside-based induction therapy."],
        [2000, "Recently, DNR has been replaced in many centers by idarubicin (IDA) as the first choice anthracycline in AML treatment."],
        [2000, "Respiratory failure during induction chemotherapy for AML (FAB M4Eo) with cytarabine and all-trans retinoic acid."],
        [2000, "We treated 53 patients with newly diagnosed AML (AML) with high-dose cytarabine-based chemotherapy followed by ATRA."],
        [2000, "arsenictrioxide has recently been used in the treatment of both relapsed and de novo AML (APML)."],
        [2000, "arsenictrioxide(arsenictrioxide) has proved highly effective in treating both refractory or primary cases of AML (APL)."],
        [2000, "arsenictrioxide, like all-trans-retinoic acid (RA), induces differentiation of AML (APL) cells in vivo."],
        [2000, "gemtuzumab-ozogamicin is a promising agent in the treatment of patients with AML that expresses CD33."],
        [2001, "Arsenic oxide (arsenictrioxide) has recently been reported to induce remission in a high percentage of patients with AML (APL)."],
        [2001, "Combination chemotherapy utilizing continuous infusion of intermediate-dose cytarabine for refractory or recurrent AML."],
        [2001, "Combined effect of all-trans retinoic acid and arsenictrioxide in AML cells in vitro and in vivo."],
        [2001, "Cytosine arabinoside, idarubicin and divided dose etoposide for the treatment of AML in elderly patients."],
        [2001, "Remission induction chemotherapy for AML typically combines cytarabine with an anthracycline or anthracycline derivative."],
        [2001, "These data suggest that cdk2 activity is likely to play a role in cytarabine-induced apoptosis in AML cells."],
        [2001, "These results suggest that dose intensification of cytarabine benefits children with AML and inv(16), as is the case in adults."],
        [2001, "arsenictrioxide (arsenictrioxide) effectively induces clinical remission via apoptosis in relapsed AML (APL)."],
        [2001, "arsenictrioxide (arsenictrioxide) induces remission in a high proportion of patients with AML (APL) via induction of apoptosis."],
        [2001, "arsenictrioxide has been shown to be effective in treating AML (APL), with minimal overall toxicity reported to date."],
        [2002, "ALL samples were significantly more sensitive than AML samples to melphalan, doxorubicin and etoposide, but not to cytarabine."],
        [2002, "CMV infection occurred rarely during cytarabine and anthracyclin based induction therapy for AML or RAEB."],
        [2002, "Compared with DS ALL, DS AML cells were significantly more sensitive to cytarabine only (21-fold)."],
        [2002, "Recent studies have shown that arsenictrioxide (As(2)O(3)) can induce complete remission in patients with AML (APL)."],
        [2002, "Resistance to cytarabine (cytarabine) is a major problem in treatment of patients with AML (AML)."],
        [2002, "Samples with chromosome 5/7 abnormalities were median 3.9-fold (P =.01) more resistant to cytarabine than other AML samples."],
        [2002, "We conclude that ATRA in combination with Ida and cytarabine can be administered safely to high-risk AML patients."],
        [2002, "arsenictrioxide (arsenictrioxide) is a novel agent to treat AML (APL)."],
        [2002, "arsenictrioxide has shown substantial efficacy in treating both newly diagnosed and relapsed patients with AML (APL)."],
        [2002, "arsenictrioxide promotes histone H3 phosphoacetylation at the chromatin of CASPASE-10 in AML cells."],
        [2003, "Aphidicolin significantly increased sensitivity to cytarabine in blast cells from both ALL (p=0.001) and AML (p0.05)."],
        [2003, "Daunorubicin (daunorubicin) is one of the most important cytotoxic agents in the treatment of AML (AML)."],
        [2003, "More comprehensive studies are needed to evaluate the apoptotic effect of dexamethasone on AML cells."],
        [2003, "Multiple complete remissions in a patient with AML (M4eo) with low-dose cytarabine and all-trans retinoic acid."],
        [2003, "The interaction of daunorubicin and mitoxantrone with the red blood cells of AML patients."],
        [2003, "arsenictrioxide (arsenictrioxide) is capable of inducing a high hematologic response rate in patients with relapsed AML (APL)."],
        [2003, "arsenictrioxide (arsenictrioxide) therapy for AML in the setting of hematopoietic stem cell transplantation."],
        [2003, "arsenictrioxide in comparison with chemotherapy and bone marrow transplantation for the treatment of relapsed AML."],
        [2003, "gemtuzumab-ozogamicin has moderate activity as a single agent in patients with CD33-positive refractory or relapsed AML (AML)."],
        [2003, "vincristine (vincristine) is an effective drug against acute lymphoblastic leukemia (ALL), many solid tumors, but not AML."],
        [2004, "5'-(3')-nucleotidase mRNA levels in blast cells are a prognostic factor in AML patients treated with cytarabine."],
        [2004, "FAK+ AML cells displayed significantly higher migration capacities and resistance to daunorubicin, compared with FAK- cells."],
        [2004, "First experience with gemtuzumab-ozogamicin plus cytarabine as continuous infusion for elderly AML patients."],
        [2004, "High-dose intermittent cytarabine is an effective postremission treatment for patients with AML (AML)."],
        [2004, "Recent resurgence in the use of arsenictrioxide is related to its high efficacy in AML (APL)."],
        [2004, "The clinical efficacy of arsenictrioxide (As(2)O(3)) has been shown in patients with relapsed AML (APL)."],
        [2004, "The combination of fludarabine, cytarabine (cytarabine) and G-CSF (FLAG) is routinely used in the treatment of AML (AML)."],
        [2004, "The impressive activity of arsenictrioxide in AML (APL) has renewed the interest in this old compound."],
        [2004, "cytarabine also induced CD80 or CD86 expression in 14 of 21 primary cultured human AML samples."],
        [2004, "mitoxantrone (MTZ) has been shown to be effective in the treatment of newly diagnosed AML (AML)."],
        [2005, "Combined treatment of AML cells by cytarabine or IDA with anti-CD33 mAb resulted in higher levels of SHP-1 phosphorylation."],
        [2005, "Cytarabine (cytarabine) is the most effective agent for the treatment of AML (AML)."],
        [2005, "Daunorubicin (daunorubicin) is commonly used to treat AML (AML)."],
        [2005, "GM-CSF and low-dose cytarabine in high-risk, elderly patients with AML or MDS."],
        [2005, "Here, azacitidine-CdR induced apoptosis in AML cells (both p53 mutant and wild-type) but not in epithelial or normal PBMCs."],
        [2005, "Intrinsically activated Ras seems to increase sensitivity of the AML blast to high-dose cytarabine therapy."],
        [2005, "Paricalcitol, when combined with arsenictrioxide, showed a markedly enhanced antiproliferative effect against AML (AML) cells."],
        [2005, "The efficacy of All-Trans Retinoic Acid (Atra) and arsenictrioxide (As(2)O(3)) in the treatment of AML (APL) is well known."],
        [2005, "Two patients with all-trans retinoic acid-resistant AML treated successfully with gemtuzumab-ozogamicin as a single agent."],
        [2005, "arsenictrioxide (As(2)O(3)) is effective against AML and has potential as a novel treatment against malignant solid tumors."],
        [2006, "Recently, patients with AML (APL) have experienced significant clinical gains after treatment with arsenictrioxide."],
        [2006, "Resistance to cytarabine (cytarabine) is a major problem in the treatment of patients with AML (AML)."],
        [2006, "Role of GSTP1-1 in mediating the effect of arsenictrioxide in the AML cell line NB4."],
        [2006, "We derived nine highly cytarabine-resistant murine BXH-2 strain AML sublines via in vitro selection."],
        [2006, "We describe 2 patients with AML (APL) in whom torsade de pointes (TdP) developed during treatment with arsenictrioxide."],
        [2006, "We have previously established the activity of clofarabine plus cytarabine in AML relapse."],
        [2006, "We searched for mechanisms of resistance in 6 patients with AML who had relapses upon midostaurin treatment."],
        [2006, "arsenictrioxide (As(2)O(3)) induces both the differentiation and apoptosis of AML cells in a concentration dependent manner."],
        [2006, "arsenictrioxide (arsenictrioxide) induces remission in patients with AML (APL)."],
        [2006, "arsenictrioxide, as a single agent, has proven efficacy in inducing molecular remission in patients with AML (APL)."],
        [2007, "All trans-retinoic acid and arsenictrioxide have already demonstrated efficacy in AML in both adults and children."],
        [2007, "Combination of all-trans-retinoic acid and gemtuzumab-ozogamicin in an elderly patient with AML and severe cardiac failure."],
        [2007, "Combined low-dose cytarabine, melphalan and mitoxantrone for older patients with AML or high-risk myelodysplastic syndrome."],
        [2007, "Results of compassionate therapy with intrathecal depot liposomalcytarabine in AML meningeosis."],
        [2007, "Reversible posterior leukoencephalopathy syndrome after repeat intermediate-dose cytarabine chemotherapy in a patient with AML."],
        [2007, "We report the results of compassionate therapy with IT depot liposomalcytarabine in 10 patients with AML with CNS involvement."],
        [2007, "arsenictrioxide (arsenictrioxide) has demonstrated effectiveness in treating AML (APL)."],
        [2007, "arsenictrioxide (arsenictrioxide) is highly efficacious in AML (APL)."],
        [2007, "arsenictrioxide (arsenictrioxide, arsenictrioxide) is used to treat patients with refractory or relapsed AML (APL)."],
        [2007, "midostaurin has proven activity in the treatment of AML (AML)."],
        [2008, "BMSCs were found to maintain cytarabine-exposed primary AML cells by protection against spontaneous apoptosis."],
        [2008, "In AML and APL cell lines, but not primary patient samples, basal catalase levels matched sensitivity to arsenictrioxide."],
        [2008, "In the last decade, arsenictrioxide (arsenictrioxide) has been used very successfully to treat AML (APL)."],
        [2008, "Inorganic arsenictrioxide (As(2)O(3)) is a highly effective treatment for AML (APL)."],
        [2008, "Previously only acute forms of leukemia particularly AML (APL) have been associated with mitoxantrone treatment in MS."],
        [2008, "arsenictrioxide (As(2)O(3)) is an effective agent for the treatment of relapsed AML (APL)."],
        [2008, "arsenictrioxide (arsenictrioxide) has been recommended for the treatment of refractory cases of AML (APL)."],
        [2008, "arsenictrioxide has remarkable efficacy in AML and is approved by the US Food and Drug Administration for this indication."],
        [2008, "gemtuzumab-ozogamicin (gemtuzumab-ozogamicin) is effective as single agent in the treatment of AML (AML)."],
        [2008, "gemtuzumab-ozogamicin (gemtuzumab-ozogamicin) monotherapy is reported to yield a 20-30% response rate in advanced AML (AML)."],
        [2009, "The standard therapeutic approaches for AML (AML) continue to be based on anthracyclines and cytarabine."],
        [2009, "The success of arsenictrioxide in the treatment of AML has renewed interest in the cellular targets of As(III) species."],
        [2009, "To report a case of isolated hyperbilirubinemia in a patient treated with cytarabine-based chemotherapy for AML."],
        [2009, "Topoisomerase IIalpha expression in AML cells that survive after exposure to daunorubicin or cytarabine."],
        [2009, "arsenictrioxide (arsenictrioxide) induces differentiation and apoptosis in AML (APL)."],
        [2009, "arsenictrioxide has been used as a therapeutic agent for AML and recently for some solid tumors."],
        [2009, "arsenictrioxide is already used clinically to treat AML demonstrating its safety profile."],
        [2009, "arsenictrioxide may reduce hyperfibrinolysis in AML by downregulation of Ann II."],
        [2009, "arsenictrioxide, As(2)O(3), has successfully been used to treat AML (APL)."],
        [2009, "sirolimus, the mTOR kinase inhibitor, sensitizes AML cells, HL-60 cells, to the cytotoxic effect of arabinozide cytarabine."],
        [2010, "It is revealed that arsenictrioxide aggravates mtDNA mutation in the D-loop region of AML cells both in vitro and in vivo."],
        [2010, "Mosaic Down syndrome-associated AML does not require high-dose cytarabine treatment for induction and consolidation therapy."],
        [2010, "Nausea and vomiting in patients with AML (AML) can be from various causes, including the use of high-dose cytarabine."],
        [2010, "Promising reports exist regarding the use of arsenictrioxide (arsenictrioxide) as first-line treatment in AML (APL)."],
        [2010, "Single cycle of arsenictrioxide-based consolidation chemotherapy spares anthracycline exposure in the primary management of AML."],
        [2010, "The expression of p38, ERK1 and Bax proteins has increased during the treatment of newly diagnosed AML with arsenictrioxide."],
        [2010, "Today, arsenictrioxide is used as one of the standard therapies for AML (APL)."],
        [2010, "arsenictrioxide (arsenictrioxide) is an effective therapeutic agent for AML (APL) and other hematopoietic malignancies."],
        [2010, "arsenictrioxide cures AML (APL) by initiating PML/RARA oncoprotein degradation, through sumoylation of its PML moiety."],
        [2010, "arsenictrioxide enhances the cytotoxic effect of thalidomide in a KG-1a human AML cell line."],
        [2011, "CD34+ AML cells are 10-15-fold more resistant to daunorubicin (daunorubicin) than CD34- AML cells."],
        [2011, "Inhibition of histone deacetylases 1 and 6 enhances cytarabine-induced apoptosis in pediatric AML cells."],
        [2011, "It is well recognized that arsenictrioxide (arsenictrioxide) is an efficacious agent for the treatment of AML (APL)."],
        [2011, "Notable success was observed in the treatment of AML (APL) with arsenictrioxide (arsenictrioxide)."],
        [2011, "The addition of arsenictrioxide to low-dose cytarabine in older patients with AML does not improve outcome."],
        [2011, "The efficacy of arsenictrioxide (arsenictrioxide) against AML (APL) and relapsed APL has been well documented."],
        [2011, "Treatment of AML cells, that have RIZ1 methylation, with azacitidine-dC, induced growth suppression with RIZ1 restoration."],
        [2011, "With the introduction of all-trans retinoic acid (ATRA) and arsenictrioxide, AML (APL) has become a highly curable malignancy."],
        [2011, "arsenictrioxide (As(2)O(3)) is an effective treatment for relapsed or refractory AML (APL)."],
        [2011, "gemtuzumab-ozogamicin combined with chemotherapy is a feasible treatment regimen in AML patients."],
        [2012, "Resistance to gemtuzumab-ozogamicin (gemtuzumab-ozogamicin) hampers the effective treatment of refractory AML (AML)."],
        [2012, "Speciation of arsenictrioxide metabolites in peripheral blood and bone marrow from an AML patient."],
        [2012, "The azacitidine/ABT-737 combination synergistically induced apoptosis in AML cells in seven of eight patients."],
        [2012, "The cytarabine (cytarabine)-based chemotherapy is the major remedial measure for AML (AML)."],
        [2012, "Upfront maintenance therapy with arsenictrioxide in AML provides no benefit for non-t(15;17) subtype."],
        [2012, "Valproic acid combined with cytarabine in elderly patients with AML has in vitro but limited clinical activity."],
        [2012, "We conclude that high BMI should not be a barrier to administer high-dose cytarabine-containing regimens for AML induction."],
        [2012, "arsenictrioxide (As(2)O(3)) is an effective therapeutic against AML and certain solid tumors."],
        [2012, "arsenictrioxide induces depolymerization of microtubules in an AML cell line."],
        [2012, "arsenictrioxide, believed to be a carcinogen and a teratogen, has found its niche in the treatment of AML (APL)."],
        [2013, "The HSP90 inhibitor NVP-AUY922-AG inhibits the PI3K and IKK signalling pathways and synergizes with cytarabine in AML cells."],
        [2013, "The efficacy of arsenictrioxide (arsenictrioxide) in the treatment of AML (APL) is widely accepted."],
        [2013, "We explored the differences by scrutinizing a case of gemtuzumab-ozogamicin (gemtuzumab-ozogamicin) in patients with AML (AML)."],
        [2013, "arsenictrioxide (arsenictrioxide) shows substantial anticancer activity in patients with AML (APL)."],
        [2013, "arsenictrioxide has been successfully used as a therapeutic in the treatment of AML (APL)."],
        [2013, "arsenictrioxide has received approval for use in patients with relapsed AML for remission induction."],
        [2013, "azacitidine and lenalidomide both have meaningful single-agent clinical activity in HR-MDS and AML with del(5q)."],
        [2013, "azacitidinecytidine treatment for relapsed or refractory AML after intensive chemotherapy."],
        [2013, "cytarabine (cytarabine or cytarabine) has been one of the cornerstones of treatment of AML since its approval in 1969."],
        [2013, "zidovudine hinders arsenictrioxide-induced apoptosis in AML cells by induction of p21 and attenuation of G2/M arrest."],
        [2014, "Adding ascorbic acid to arsenictrioxide produces limited benefit in patients with AML excluding AML."],
        [2014, "Knockdown of CD44 enhances chemosensitivity of AML cells to ADM and cytarabine."],
        [2014, "Limited data are available on azacitidine (azacitidine) treatment and its prognostic factors in AML (AML)."],
        [2014, "Our study provides a novel clue that CD44 plays a significant role in the chemoresistance of AML cells to cytarabine and ADM."],
        [2014, "The experience with gemtuzumab-ozogamicin has highlighted both the potential value and limitations of antibodies in AML (AML)."],
        [2014, "Thioredoxin-1 inhibitor PX-12 induces human AML cell apoptosis and enhances the sensitivity of cells to arsenictrioxide."],
        [2014, "arsenictrioxide (arsenictrioxide) is a novel form of therapy that has been found to aid AML (APL) patients."],
        [2014, "arsenictrioxide (arsenictrioxide) is a promising antitumor agent used to treat AML (APL) and, recently solid tumor."],
        [2014, "cAMP protects AML cells from arsenictrioxide-induced caspase-3 activation and apoptosis."],
        [2014, "gemtuzumab-ozogamicin was the first example of antibody-directed chemotherapy in cancer, and was developed for AML."],
        [2015, "Finally, Msi2 silencing in AML cells also enhanced their chemosensitivity to daunorubicin."],
        [2015, "To highlight the acceptable results seen after use of low dose cytarabine in elderly patients of AML (AML) with comorbidities."],
        [2015, "arsenictrioxide (As2 O3 ) is commonly used to treat AML and solid tumors."],
        [2015, "arsenictrioxide has been successfully used for the treatment of patients with AML (APL) worldwide."],
        [2015, "arsenictrioxide is an effective and potent antitumor agent used in patients with AML and produces dramatic remissions."],
        [2015, "azacitidine seems a reasonable therapeutic option for most unfit AML patients, i.e."],
        [2015, "azacitidine sensitization to arsenictrioxide treatment was re-capitulated also in primary AML samples."],
        [2015, "azacitidine sensitizes AML cells to arsenictrioxide by up-regulating the arsenic transporter aquaglyceroporin 9."],
        [2015, "gemtuzumab-ozogamicin in combination with intensive chemotherapy in relapsed or refractory AML."],
        [2015, "high-dose cytarabine therapy followed by allo-HSCT could improve the prognosis of MK-positive AML patients."],
        [2016, "Superior anti-tumor activity of the MDM2 antagonist idasanutlin and the Bcl-2 inhibitor venetoclax in p53 wild-type AML models."],
        [2016, "These results highly suggest that the pharmacogenetic analyses of cytarabine influx may be decisive in AML patients."],
        [2016, "Three patients with high-grade myelodysplastic syndrome or AML who received azacitidine."],
        [2016, "Thus NVP-AUY922 and cytarabine combination therapy might be a prospective strategy for AML treatment."],
        [2016, "Venetoclax demonstrated activity and acceptable tolerability in patients with AML and adverse features."],
        [2016, "and abrogated resistance to cytarabine in AML cells cocultured with HS-5 stromal cells."],
        [2016, "arsenictrioxide (arsenictrioxide) has demonstrated clinical efficacy in AML (APL) and in vitro activity in various solid tumors."],
        [2016, "arsenictrioxide (arsenictrioxide) is an efficient drug for the treatment of the patients with AML (APL)."],
        [2016, "cytarabine, sorafenib) in all tested AML cell lines regardless of their FLT3 mutation status."],
        [2016, "miR-29c is of prognostic value and influences response to azacitidine treatment in older AML patients."],
        [2017, "Enasidenib induces AML cell differentiation to promote clinical response."],
        [2017, "Intensive combination chemotherapy including gemtuzumab-ozogamicin emerged as an effective salvage therapy in refractory AML."],
        [2017, "Synergistic antileukemic interactions between AZ20 and cytarabine were confirmed in primary AML patient samples."],
        [2017, "The FDA has approved the small-molecule inhibitor midostaurin in combination with chemotherapy to treat AML."],
        [2017, "arsenictrioxide (arsenictrioxide) is an old drug that has recently been reintroduced as a therapeutic agent for AML (APL)."],
        [2017, "arsenictrioxide (arsenictrioxide) is highly effective in the treatment of patients with AML (APL)."],
        [2017, "combination effects of arsenictrioxide and sorafenib in two AML cell lines, KG-1 and U937."],
        [2017, "cytarabine (cytarabine) is one of the key drugs for treating AML (AML)."],
        [2017, "cytarabine (cytarabine) remains the backbone of most treatment regimens for AML (AML)."],
        [2017, "cytarabine-resistant AML cell lines were sensitive to BTZ and DSF/Cu2+."],
        [2018, "The current standard of care for the treatment of patients with newly diagnosed AML (AML) is an anthracycline plus cytarabine."],
        [2018, "The dose escalation of cytarabine in induction therapy lead to improved remission rates in the elderly AML patients."],
        [2018, "The ectopic expression of hTERT significantly attenuated the apoptotic effect of midostaurin on AML cells."],
        [2018, "The proportion of patients with AML (AML) cured is increased by administering high-dose cytarabine (HiDAC)."],
        [2018, "Venetoclax (venetoclax) targets BCL-2 and has shown promising efficacy in AML but over-expression of MCL-1 can cause resistance."],
        [2018, "We examined the role of TET-FOXP3 axis in the cytotoxic effects of arsenictrioxide on the human AML cell line, U937."],
        [2018, "arsenictrioxide (As₂O₃), a traditional remedy in Chinese medicine, has been used in AML (APL) research and clinical treatment."],
        [2018, "arsenictrioxide enhance reactive oxygen species levels and induce apoptosis and suppresses proliferation in AML cells."],
        [2018, "arsenictrioxide promoting ETosis in AML through mTOR-regulated autophagy."],
        [2018, "miR-134 increases the antitumor effects of cytarabine by targeting Mnks in AML cells."],
        [2019, "Furthermore, a potentiating effect of HHT on arsenictrioxide was also observed in primary AML cells and AML xenografted tumors."],
        [2019, "MALAT1 knockdown inhibits proliferation and enhances cytarabine chemosensitivity by upregulating miR-96 in AML cells."],
        [2019, "Using a genome-wide CRISPR/Cas9 screen in human AML, we identified genes whose inactivation sensitizes AML blasts to venetoclax."],
        [2019, "Venetoclax is approved for older untreated AML (AML) patients."],
        [2019, "We found that venetoclax and voreloxin synergistically induced apoptosis in multiple AML cell lines."],
        [2019, "arsenictrioxide (arsenictrioxide) has been used clinically for the treatment of AML and some solid tumors."],
        [2019, "arsenictrioxide and all-trans retinoic acid have become the frontline treatments for patients with AML (APL)."],
        [2019, "cytarabine is the clinically most relevant cytotoxic agent for AML treatment."],
        [2019, "gemtuzumab-ozogamicin was approved for the treatment of AML by the US Food and Drug Administration in September 2017."],
        [2019, "omacetaxinemepesuccinate potentiates the antileukemic activity of arsenictrioxide against AML cells."],
        [2020, "Abivertinib synergistically strengthens the anti-leukemia activity of venetoclax in AML in a BTK-dependent manner."],
        [2020, "FHL1-targeted intervention enhances the sensitivity of AML cells to cytarabine."],
        [2020, "Gilteritinib in the treatment of relapsed and refractory AML with a FLT3 mutation."],
        [2020, "Gilteritinib is a FLT3 kinase inhibitor approved for FLT3-mutated AML (AML)."],
        [2020, "Venetoclax combined with 5 + 2 induction chemotherapy was safe and tolerable in fit older patients with AML."],
        [2020, "Venetoclax is a highly selective BCL-2 inhibitor that has been approved by the FDA for treating elderly AML patients."],
        [2020, "Venetoclax-based therapy can induce responses in approximately 70% of older previously untreated patients with AML (AML)."],
        [2020, "arsenictrioxide (arsenictrioxide) has been proved useful for the treatment of AML (APL)."],
        [2020, "arsenictrioxide (arsenictrioxide) is one of the most effective drugs for treatment of AML (APL)."],
        [2020, "arsenictrioxide induces autophagic degradation of the FLT3-ITD mutated protein in FLT3-ITD AML cells."],
        [2021, "225Ac-labeled CD33-targeting antibody reverses resistance to Bcl-2 inhibitor venetoclax in AML models."],
        [2021, "All-trans retinoic acid (ATRA) and pre-upfront arsenictrioxide (arsenictrioxide) have revolutionized the therapy of AML (APL)."],
        [2021, "Analysis of gamma-glutamyltransferase in AML patients undergoing arsenictrioxide treatment."],
        [2021, "CD44 loss of function sensitizes AML cells to the BCL-2 inhibitor venetoclax by decreasing CXCL12-driven survival cues."],
        [2021, "Cytarabine and daunorubicin are old drugs commonly used in the treatment of AML (AML)."],
        [2021, "EVs derived from both new cases and relapsed AML patients significantly reduced idarubicin-induced apoptosis in the U937 cells."],
        [2021, "Enhanced cytarabine-induced killing in OGG1-deficient AML cells."],
        [2021, "For decades two chemotherapeutic agents, cytarabine and daunorubicin, remained the backbone of AML therapy protocols."],
        [2021, "Furthermore, venetoclax (a BCL2 inhibitor) synergistically enhanced the cytotoxicity of DNR in AML cell lines."],
        [2021, "HDAC8 promotes daunorubicin resistance of human AML cells via regulation of IL-6 and IL-8."],
        [2022, "Acquired genetic mutations can confer resistance to arsenictrioxide (arsenictrioxide) in the treatment of AML (APL)."],
        [2022, "Additionally, arsenictrioxide provoked specific cytoprotective effects in the AML cell lines HL-60 and U937."],
        [2022, "Azacitidine-induced reconstitution of the bone marrow T cell repertoire is associated with superior survival in AML patients."],
        [2022, "BCL-2 inhibition has been shown to be effective in AML (AML) in combination with hypomethylating agents or low-dose cytarabine."],
        [2022, "By coculture with cells, this study revealed that the AML cells resisted apoptosis induced by the anticancer drug cytarabine."],
        [2022, "Combination strategies to promote sensitivity to cytarabine-induced replication stress in AML with and without DNMT3A mutations."],
        [2022, "Downregulation of MSK1 enhanced the sensitivity of AML cells to cytarabine."],
        [2022, "Epigenetic therapy with chidamide alone or combined with 5-azacitidine exerts antitumour effects on AML cells in vitro."],
        [2022, "Furthermore, we discovered that inhibiting SPAG1 impacted AML cell susceptibility to venetoclax."],
        [2022, "Gilteritinib is approved for relapsed or refractory FLT3 mutated AML as monotherapy based on the ADMIRAL study."],
    ]
    
    #df = pd.read_csv('https://docs.google.com/spreadsheets/d/' + '1A-O505D_vkOprtWKMyQ1hIgNlaLwwBZnIZukiiVTncQ' + '/export?gid=0&format=csv', sep=',', escapechar='\\')
    
    return pd.DataFrame(data, columns=['filename', 'sentences'])

def flat_list(composed_list):
    if any(isinstance(x, list) for x in composed_list):
        composed_list = [item for sublist in composed_list for item in sublist]

    return composed_list

# MAIN PROGRAM:
if __name__ == '__main__':
    if 'execution_counter' not in st.session_state:
        st.session_state['execution_counter'] = 0
        
    sentences_df = get_sentences_dataset()
        
    hide_streamlit_style = """
            <style>           
            footer {
                visibility: hidden;
            }
            
            footer:after {
                content:'Developed by Matheus Vargas Volpon Berto.'; 
                visibility: visible;
                display: block;
                position: relative;
                padding: 5px;
                top: 2px;
                color: black;
            }
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    with st.sidebar.form('sidebar_form'):
        st.header('Models exploration settings')
        
        loaded_model = st.selectbox(
         'Choose one of the preloaded models:',
         ('19: 1921 - 2022',
          '18: 1921 - 2018',
          '17: 1921 - 2014',
          '16: 1921 - 2013',
          '15: 1921 - 2011',
          '14: 1921 - 2009',
          '13: 1921 - 2001',
          '12: 1921 - 1999',
          '11: 1921 - 1998',
          '10: 1921 - 1995',
          '09: 1921 - 1983',
          '08: 1921 - 1982',
          '07: 1921 - 1977',
          '06: 1921 - 1976',
          '05: 1921 - 1974',
          '04: 1921 - 1971',
          '03: 1921 - 1969',
          '02: 1921 - 1967',
          '01: 1921 - 1963'))
        
        top_n = st.slider('Select the neighborhood size',
            5, 20, (5), 5)
        
        input_sentence = st.text_input(label='Input sentence', max_chars=128, help='Type the sentence that you want to compare to the others.')
        
        submitted = st.form_submit_button('Apply settings')
    
    st.sidebar.header('GitHub Repository')
    st.sidebar.markdown("[![Foo](https://cdn-icons-png.flaticon.com/32/25/25231.png)](https://github.com/matheusvvb-19/WE4LKD-leukemia_w2v)")
        
    st.title('Sentence Viewer')
    st.header('Sentence Embedding Visualization Based on Cosine Similarity')
    with st.expander('How to use this app'):
        st.markdown('**Sidebar**')
        st.markdown('Select the BERT-based model that you want to explore. Then, define the number of most similar sentences from the dataset that you want to compare to your input sentence. Finally, type your input sentence in the text box and click on "Apply settings"')

        st.markdown('**Main window**')
        st.markdown('_Hint: To see this window content better, you can minimize the sidebar._')
        st.markdown('lalala')
        
    if submitted or st.session_state['execution_counter'] != 0:
        st.markdown('**Input**')
        st.markdown(input_sentence)
        
        st.session_state['execution_counter'] += 1

        aux_df = sentences_df[(sentences_df["filename"].values <= int(loaded_model[-4:]))]
        sentences = aux_df['sentences'].to_list()
        sentences.insert(0, input_sentence)

        tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        model = AutoModel.from_pretrained('matheusvolpon/WE4LKD_AML_distilbert_1921_{}'.format(loaded_model[-4:]))

        # initialize dictionary that will contain tokenized sentences
        tokens = {'input_ids': [], 'attention_mask': []}

        for sentence in sentences:
            # tokenize sentence and append to dictionary lists
            new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True,
                                               padding='max_length', return_tensors='pt')
            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])

        # reformat list of tensors into single tensor
        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

        outputs = model(**tokens)

        embeddings = outputs.last_hidden_state

        attention_mask = tokens['attention_mask']

        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask

        # convert from PyTorch tensor to numpy array
        mean_pooled = mean_pooled.detach().numpy()

        # calculate
        similarities = flat_list(cosine_similarity([mean_pooled[0]], mean_pooled[1:]).tolist())
        
        data = {
            'sentence': [],
            'similarity': [],
        }

        for s, si in zip(sentences[1:], similarities):
            data['sentence'].append(s)
            data['similarity'].append(si)

        df_similar_sentences = pd.DataFrame(data).sort_values(by=['similarity'], ascending=False)
        
        st.markdown('**Top {} similar sentences to the input**'.format(top_n))
        st.table(df_similar_sentences.head(top_n))
