import settings
import sets

settings = settings.Settings()
train_set = sets.TrainSet(settings)
comb_gen_test_set = sets.TestSet(settings, test_type = "comb_gen")
anti_test_set = sets.TestSet(settings, test_type = "anti")
i, t = comb_gen_test_set.get_exemplar()
print(i)
print(t)
# print(comb_gen_test_set.get_exemplar())
# print(anti_test_set.get_exemplar())