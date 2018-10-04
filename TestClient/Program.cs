using System;
using System.Collections.Generic;
using System.Linq;

namespace TestClient
{
    public interface ITransform
    {
    }

    public interface ITransform<TIn, TOut> : ITransform
    {
        TOut Transform(TIn corpus);
    }

    public interface IMetadata
    {
    }

    public interface IMetadata<TOut>
    {
        TOut GetMetadata();
    }

    public interface IFeaturize : ITransform, IMetadata
    {
    }

    public interface IPipeline
    {
        void AddTransformer(string name, ITransform transformer);
        void AddFeaturizer(string name, IFeaturize featurizer);
    }




    public class Tokenizer : ITransform<IEnumerable<string>, IEnumerable<IEnumerable<string>>>
    {
        private IEnumerable<string> Tokenize(string doc)
        {
            return doc.ToLower().Split(new char[] { ' ', ',', '.', '?', '!' }, StringSplitOptions.RemoveEmptyEntries);
        }

        public IEnumerable<IEnumerable<string>> Transform(IEnumerable<string> corpus)
        {
            return corpus.Select(d => Tokenize(d));
        }
    }




    public class TFIDFBOW : ITransform<IEnumerable<IEnumerable<string>>, IEnumerable<IEnumerable<float>>>, IMetadata<Dictionary<string, int>>
    {
        private Dictionary<string, (int, Dictionary<int, int>)> bow = new Dictionary<string, (int, Dictionary<int, int>)>();  // token : (freq in dataset, (doc, [freq in doc]))

        static float TFIDF(float numberOfDocs, float datasetFreq, float docFreq)
        {
            float idf = (float)Math.Log(numberOfDocs / ((float)1 + docFreq));
            return docFreq * idf;
        }

        public IEnumerable<IEnumerable<float>> Transform(IEnumerable<IEnumerable<string>> corpusTokenized)
        {
            //Create Temp BOW Dictionary with word frequency in dataset and word frequency per document
            bow = new Dictionary<string, (int, Dictionary<int, int>)>();  // token : (freq in dataset, (doc, [freq in doc]))
            var docId = 0;
            foreach (var doc in corpusTokenized)
            {
                var wordsAlreadyInDoc = new List<string>();

                foreach (var word in doc)
                {
                    if (bow.ContainsKey(word))
                    {
                        var x = bow[word];
                        x.Item1++;
                        if (wordsAlreadyInDoc.Contains(word))
                        {
                            x.Item2[docId] += 1;
                        }
                        else
                        {
                            x.Item2[docId] = 1;
                        }
                        bow[word] = x;
                    }
                    else
                    {
                        bow[word] = (1, new Dictionary<int, int>() { { docId, 1 } });
                    }


                    wordsAlreadyInDoc.Add(word);
                }

                docId++;
            }

            //Create Features Vectors with BOW TF-IDF values
            var vectors = new List<List<float>>();
            docId = 0;
            foreach (var doc in corpusTokenized)
            {
                vectors.Add(new List<float>());
                foreach (KeyValuePair<string, (int, Dictionary<int, int>)> bowToken in bow)
                {
                    float tfidf = 0;

                    if (doc.Contains(bowToken.Key))
                    {
                        tfidf = TFIDF(corpusTokenized.Count(), bowToken.Value.Item1, bowToken.Value.Item2[docId]);
                    }

                    vectors.Last().Add(tfidf);
                }

                docId++;
            }

            //Normalize Vectors using L2
            foreach (var vector in vectors)
            {
                float SqrtSumSquared = (float)Math.Sqrt(vector.Aggregate((sum, value) => sum + (value * value)));

                for (int i = 0; i < vector.Count(); i++)
                {
                    vector[i] /= SqrtSumSquared;
                }

            }

            return vectors;
        }

        public Dictionary<string, int> GetMetadata() // token : freq in dataset
        {
            return bow.ToDictionary(arg => arg.Key, arg => arg.Value.Item1);
        }
    }






    class Program
    {
        static string[] dataset = { "Well done! You really made a great job. Really well done!!",
                                    "Good work.I appreciate your effort",
                                    "Great effort. You should continue this way",
                                    "nice work man, that was great",
                                    "Excellent job! vEry well."
                                  };


        static void Main(string[] args)
        {
            var tokenizer = new Tokenizer();
            var tokens = tokenizer.Transform(dataset);

            var bower = new TFIDFBOW();
            var vectors = bower.Transform(tokens);
            var bow = bower.GetMetadata();

            Console.WriteLine(vectors);
            Console.WriteLine(bow);
        }
    }
}
