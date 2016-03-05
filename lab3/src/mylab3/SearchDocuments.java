/**************************************************************************
 * Author: Xu Chencan
 * Date: 4/4/2015
 * Problem: Given a set of keywords, find the top 3 most similar 
 * documents among a set of N documents using the TF*IDF metric
 * Input file: query.txt, sw3.txt, documents(f1.txt,...,f10.txt) to search
 * Output: the top 3 most similar documents
 * Method: Five MapReduce Stages:
 *         1) Compute frequency of every word in a document
 *         2) Compute tf-idf of every word w.r.t. a document
 *         3) Compute normalized tf-idf
 *         4) Compute the relevance of every document w.r.t. a query
 *         5) Get topk documents
 *
 *Note: Remove stop-words like “a”, “the”, “that”, “of”. 
 *************************************************************************/
package mylab3;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeMap;

import org.apache.commons.collections.IteratorUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class SearchDocuments {

	// Stage 1: Compute frequency of every word in a document
	// Mapper 1: (tokenize file)
	public static class TokenizerMapper extends
			Mapper<Object, Text, Text, IntWritable> {

		Set<String> stopwords = new HashSet<String>();

		@Override
		protected void setup(Context context) {
			try {
				Path path = new Path("/search/sw3.txt");
				FileSystem fs = FileSystem.get(new Configuration());
				BufferedReader br = new BufferedReader(new InputStreamReader(
						fs.open(path)));
				String word = null;
				while ((word = br.readLine()) != null) {
					stopwords.add(word);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		private Text word_filename = new Text();
		private final static IntWritable one = new IntWritable(1);

		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {

			// Get file name for a key/value pair in the Map function
			String filename = ((FileSplit) context.getInputSplit()).getPath()
					.getName();

			// read one line. tokenize into (word@filename, 1) pairs
			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				String word = itr.nextToken();
				if (!stopwords.contains(word)) {
					word_filename.set(word + "@" + filename);
					context.write(word_filename, one);
				}
			}
		}
	}

	// Reducer 1: (calculate frequency of every word in every file)
	public static class IntSumReducer extends
			Reducer<Text, IntWritable, Text, IntWritable> {

		private IntWritable result = new IntWritable();

		public void reduce(Text key, Iterable<IntWritable> values,
				Context context) throws IOException, InterruptedException {
			// sum up all the values, output (word@filename, freq) pair
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			result.set(sum);
			context.write(key, result);
		}
	}

	// Stage 2: Compute tf-idf of every word w.r.t. a document
	// Mapper 2:parse the output of stage1
	public static class Mapper2 extends Mapper<Object, Text, Text, Text> {

		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {
			// parse the key/value pair into word, filename, frequency
			String word_filename = key.toString();
			String[] wordandfilename = word_filename.split("@");
			String word = wordandfilename[0];
			String filename = wordandfilename[1];
			String frequency = value.toString();
			// output a pair (word, filename=frequency)
			context.write(new Text(word), new Text(filename + "=" + frequency));
		}
	}

	// Reducer 2: (calculate tf-idf of every word in every document)
	public static class Reducer2 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			// Note: key is a word, values are in the form of
			// (filename=frequency)
			String word = key.toString();
			// sum up the number of files containing a particular word
			ArrayList<String> valuelist = new ArrayList<String>();
			for (Text val : values) {
				valuelist.add(val.toString());
			}

			int sum = valuelist.size();
			// for every filename=frequency in the value, compute tf-idf of this
			// word in filename and output (word@filename, tfidf)
			int N = 10; // the total number of documents
			for (int i = 0; i < sum; i++) {
				String[] filenameandfrequency = valuelist.get(i).split("=");
				String filename = filenameandfrequency[0];
				int frequency = Integer.valueOf(filenameandfrequency[1]);
				double tfidf = (1 + Math.log(1.0 * frequency))
						* Math.log(1.0 * N / sum);
				context.write(new Text(word + "@" + filename),
						new Text(String.valueOf(tfidf)));
			}
		}
	}

	// Stage 3: Compute normalized tf-idf
	// Mapper 3: (parse the output of stage 2)
	public static class Mapper3 extends Mapper<Object, Text, Text, Text> {
		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {
			// parse the key/value pair into word, filename, tfidf
			String[] wordandfilename = key.toString().split("@");
			String word = wordandfilename[0];
			String filename = wordandfilename[1];
			String tfidf = value.toString();
			// output a pair(filename, word=tfidf)
			context.write(new Text(filename), new Text(word + "=" + tfidf));
		}
	}

	// Reducer 3: (compute normalized tf-idf of every word in very document)
	public static class Reducer3 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			// Note: key is a filename, values are in the form of (word=tfidf)
			String filename = key.toString();

			ArrayList<String> valuelist = new ArrayList<String>();
			for (Text val : values) {
				valuelist.add(val.toString());
			}
			double squaresum = 0;
			for (int i = 0; i < valuelist.size(); i++) {
				String val = valuelist.get(i);
				String[] wordandtfidf = val.split("=");
				double tfidf = Double.valueOf(wordandtfidf[1]);
				squaresum += tfidf * tfidf;
			}
			// for every word=tfidf in the value, output (word@filename,
			// norm-tfidf)

			for (int i = 0; i < valuelist.size(); i++) {
				String val = valuelist.get(i);
				String[] wordandtfidf = val.split("=");
				String word = wordandtfidf[0];
				double tfidf = Double.valueOf(wordandtfidf[1]);
				double normtfidf = tfidf / Math.sqrt(squaresum);
				context.write(new Text(word + "@" + filename),
						new Text(String.valueOf(normtfidf)));
			}
		}
	}

	// Stage 4: Compute the relevance of every document w.r.t. a query
	// Mapper 4: (parse the output of stages)
	public static class Mapper4 extends Mapper<Object, Text, Text, Text> {

		HashSet<String> query = new HashSet<String>();

		@Override
		protected void setup(Context context) {
			try {
				Path path = new Path("/search/query.txt");
				FileSystem fs = FileSystem.get(new Configuration());
				BufferedReader br = new BufferedReader(new InputStreamReader(
						fs.open(path)));
				StringTokenizer itr = new StringTokenizer(br.readLine());
				while (itr.hasMoreTokens()) {
					String word = itr.nextToken();
					query.add(word);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {
			// parse the key/value pair into word, filename, norm-tfidf
			String[] wordandfilename = key.toString().split("@");
			String word = wordandfilename[0];
			String filename = wordandfilename[1];
			String normtfidf = value.toString();
			// if the word is contained in the query file, output (filename,
			// word=norm-tfidf)
			if (query.contains(word)) {
				context.write(new Text(filename), new Text(word + "="
						+ normtfidf));
			}
		}
	}

	// Reducer 4: (calculate relevance of every document w.r.t. the query)
	public static class Reducer4 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			// Note: key is a filename, values are in the form of
			// (word=norm-tfidf)
			String filename = key.toString();
			double totalnormtfidf = 0.0;
			for (Text val : values) {
				String[] wordandnormtfidf = val.toString().split("=");
				double normtfidf = Double.valueOf(wordandnormtfidf[1]);
				totalnormtfidf += normtfidf;
			}
			context.write(new Text(filename),
					new Text(String.valueOf(totalnormtfidf)));
		}
	}

	// Stage 5: Get topk documents
	// Get local Top K
	public static class TopKMapper extends
			Mapper<Object, Text, NullWritable, Text> {
		// if the norm-tfidf of two documents are the same, both of them will be
		// recorded in the map
		private TreeMap<Double, ArrayList<String>> tmap = null;
		private ArrayList<String> filenames = null;
		private NullWritable nwkey = NullWritable.get();

		public void setup(Context context) throws IOException,
				InterruptedException {
			// initialize tmap for current map
			tmap = new TreeMap<Double, ArrayList<String>>();
		}

		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {
			// get filename and its relevance
			String filename = key.toString();
			double relevance = Double.valueOf(value.toString());

			// if the relevance is in tmap, the add the word into its Arraylist
			if (tmap.containsKey(relevance)) {
				filenames = tmap.get(relevance);
				filenames.add(filename);
			}
			// otherwise, insert the filename into a new ArrayList(flist)
			// and put <relevance, flist> into tmap
			else {
				filenames = new ArrayList<String>();
				filenames.add(filename);
				tmap.put(relevance, filenames);
			}
			// if tmap contains more than 3 pairs, remove the pair with the
			// smallest frequency
			if (tmap.size() > 3) {
				tmap.remove(tmap.firstKey());
			}
		}

		public void cleanup(Context context) throws IOException,
				InterruptedException {
			// transform all the entries in tmap into key/value pairs as
			// mapper's output.
			// note: need to extract all the words form each ArrayList.
			// note: as we want to use one reduce task, all the pairs must
			// have the same key (e.g., NullWritable), and the value is a
			// combination of relevance and filename (e.g. "relevance:filename")
			for (Double relevance : tmap.keySet()) {
				filenames = tmap.get(relevance);
				for (String filename : filenames) {
					context.write(nwkey, new Text(String.valueOf(relevance)
							+ ":" + filename));
				}
			}

		}
	}

	// Get Global Top K
	public static class TopKReducer extends
			Reducer<NullWritable, Text, Text, Text> {

		private TreeMap<Double, ArrayList<String>> tmap = null;

		public void setup(Context context) throws IOException,
				InterruptedException {
			tmap = new TreeMap<Double, ArrayList<String>>();

		}

		public void reduce(NullWritable key, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {

			double relevance = 0.0;
			String filename = null;
			String[] relevanceandfilename = new String[2];
			ArrayList<String> filenames = null;
			for (Text val : values) {
				// parse values into <relevance, filename> pairs and put them
				// into
				// TreeMap(tmap) one by one.
				relevanceandfilename = val.toString().split(":");
				relevance = Double.valueOf(relevanceandfilename[0]);
				filename = relevanceandfilename[1];

				if (tmap.containsKey(relevance)) {
					filenames = tmap.get(relevance);
					filenames.add(filename);
				} else {
					filenames = new ArrayList<String>();
					filenames.add(filename);
					tmap.put(relevance, filenames);
				}
				// Whenever there are more than 3 pairs
				// in tmap, remove one pair with the smallest frequency. The
				// operations here are similar to those in the map function.
				if (tmap.size() > 3) {
					tmap.remove(tmap.firstKey());
				}
			}
		}

		public void cleanup(Context context) throws IOException,
				InterruptedException {
			// output 3 pairs in TreeMap(tmap) reversely

			// copy treemap into navigablemap in a reverse order of keys
			NavigableMap<Double, ArrayList<String>> reverseTreeMap = tmap
					.descendingMap();
			int count = 0;
			for (Map.Entry<Double, ArrayList<String>> entry : reverseTreeMap
					.entrySet()) {
				double relevance = entry.getKey();
				ArrayList<String> filenames = entry.getValue();
				for (String filename : filenames) {
					context.write(new Text(String.valueOf(relevance)),
							new Text(filename));
					count++;
				}
				if (count >= 3)
					break;

			}
		}
	}

	public static void main(String[] args) throws IOException,
			InterruptedException, ClassNotFoundException {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args)
				.getRemainingArgs();
		if (otherArgs.length != 6) {
			System.err
					.println("Usage: searchdocuments <documents directory> <output1> "
							+ "<output2> <output3> <output4> <output5>");
			System.exit(2);
		}

		// Stage 1: Compute frequency of every word in a document
		Job job1 = new Job(conf, "word-filename count");
		job1.setJarByClass(SearchDocuments.class);
		job1.setMapperClass(TokenizerMapper.class);
		job1.setCombinerClass(IntSumReducer.class);
		job1.setReducerClass(IntSumReducer.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(IntWritable.class);
		job1.setNumReduceTasks(5);
		FileInputFormat.addInputPath(job1, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job1, new Path(otherArgs[1]));
		job1.waitForCompletion(true);

		// Stage 2: Compute tf-idf of every word w.r.t. a document
		Job job2 = new Job(conf, "calculate tfidf");
		job2.setJarByClass(SearchDocuments.class);
		job2.setInputFormatClass(KeyValueTextInputFormat.class);
		FileInputFormat.addInputPath(job2, new Path(otherArgs[1]));
		job2.setMapperClass(Mapper2.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(Text.class);

		job2.setReducerClass(Reducer2.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(Text.class);
		job2.setNumReduceTasks(4);
		FileOutputFormat.setOutputPath(job2, new Path(otherArgs[2]));
		job2.waitForCompletion(true);

		// Stage 3: Compute normalized tf-idf
		Job job3 = new Job(conf, "nomalize tfidf");
		job3.setJarByClass(SearchDocuments.class);
		job3.setInputFormatClass(KeyValueTextInputFormat.class);
		FileInputFormat.addInputPath(job3, new Path(otherArgs[2]));
		job3.setMapperClass(Mapper3.class);
		job3.setMapOutputKeyClass(Text.class);
		job3.setMapOutputValueClass(Text.class);

		job3.setReducerClass(Reducer3.class);
		job3.setOutputKeyClass(Text.class);
		job3.setOutputValueClass(Text.class);
		job3.setNumReduceTasks(3);
		FileOutputFormat.setOutputPath(job3, new Path(otherArgs[3]));
		job3.waitForCompletion(true);

		// Stage 4: Compute the relevance of every document w.r.t. a query
		Job job4 = new Job(conf, "compute revelance");
		job4.setJarByClass(SearchDocuments.class);
		job4.setInputFormatClass(KeyValueTextInputFormat.class);
		FileInputFormat.addInputPath(job4, new Path(otherArgs[3]));
		job4.setMapperClass(Mapper4.class);
		job4.setMapOutputKeyClass(Text.class);
		job4.setMapOutputValueClass(Text.class);

		job4.setReducerClass(Reducer4.class);
		job4.setOutputKeyClass(Text.class);
		job4.setOutputValueClass(Text.class);
		job4.setNumReduceTasks(2);
		FileOutputFormat.setOutputPath(job4, new Path(otherArgs[4]));
		job4.waitForCompletion(true);

		// Stage 5: Get topk documents
		Job job5 = new Job(conf, "get top K documents");
		job5.setJarByClass(SearchDocuments.class);
		job5.setInputFormatClass(KeyValueTextInputFormat.class);
		FileInputFormat.addInputPath(job5, new Path(otherArgs[4]));
		job5.setMapperClass(TopKMapper.class);
		job5.setMapOutputKeyClass(NullWritable.class);
		job5.setMapOutputValueClass(Text.class);

		job5.setReducerClass(TopKReducer.class);
		job5.setOutputKeyClass(Text.class);
		job5.setOutputValueClass(Text.class);

		job5.setNumReduceTasks(1);

		FileOutputFormat.setOutputPath(job5, new Path(otherArgs[5]));
		job5.waitForCompletion(true);

	}

}
