/*******************************************************************
 * Author: Xu Chencan
 * Date: 3/18/2015
 * Problem: Given two textual files, count the number of words that 
 * are common. Return the top 20 answers. Use no more that 2 stages.
 * Scenarios to consider: remove stop words in "sw3.txt"
 * Input file: 844.txt and 1952.txt
 * Output: top 20 common words and their frequency
 * Method: Two MapReduce Stages 1) count common words
 *                              2) get top 20 common words
 *
 *Note: Use combiner to do optimization. 
 *******************************************************************/

package mylab2;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeMap;

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
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class TopKCommonWordsO {

	// tokenize file 1
	public static class TokenizerWCMapper1 extends
			Mapper<Object, Text, Text, Text> {

		Set<String> stopwords = new HashSet<String>();

		@Override
		protected void setup(Context context) {
			try {
				Path path = new Path("/data/input3/sw3.txt");
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

		private Text word = new Text();
		private final static Text identifier = new Text("f1");

		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				word.set(itr.nextToken());
				if (stopwords.contains(word.toString()))
					continue;
				context.write(word, identifier);
			}
		}
	}

	// tokenize file 2
	public static class TokenizerWCMapper2 extends
			Mapper<Object, Text, Text, Text> {

		Set<String> stopwords = new HashSet<String>();

		@Override
		protected void setup(Context context) {
			try {
				Path path = new Path("/data/input3/sw3.txt");
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

		private Text word = new Text();
		private final static Text identifier = new Text("f2");

		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				word.set(itr.nextToken());
				if (stopwords.contains(word.toString()))
					continue;
				context.write(word, identifier);
			}
		}
	}

	// use combiner
	public static class WCCombiner extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			// initialize two counts count1 and count2 with values of 0.
			int count1 = 0;
			int count2 = 0;

			// parse each value into <file identifier, frequency> pairs, if
			// identifier="f1", then increase count1 by frequency; if
			// identifier="f2", then increase count2 by frequency.
			for (Text val : values) {
				String identifier = val.toString();
				if (identifier.equals("f1")) {
					count1++;
				} else if (identifier.equals("f2")) {
					count2++;
				}
			}
			context.write(key, new Text("f1_" + String.valueOf(count1)));
			context.write(key, new Text("f2_" + String.valueOf(count2)));
		}
	}

	// get the number of common words
	public static class CommonWordsReducer extends
			Reducer<Text, Text, Text, IntWritable> {

		private IntWritable commoncount = new IntWritable();

		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			int count1 = 0;
			int count2 = 0;
			String[] identifierandfrequency = new String[2];
			for (Text val : values) {
				identifierandfrequency = val.toString().split("_");
				if (identifierandfrequency[0].equals("f1")) {
					count1 += Integer.valueOf(identifierandfrequency[1]);
				} else if (identifierandfrequency[0].equals("f2")) {
					count2 += Integer.valueOf(identifierandfrequency[1]);
				}
			}
			if (count1 != 0 && count2 != 0) {
				if (count1 > count2) {
					count1 = count2;
				}
				commoncount.set(count1);
				context.write(key, commoncount);
			}
		}
	}

	// Get local Top K
	public static class TopKMapper extends
			Mapper<Object, Text, NullWritable, Text> {
		// if two words appear the same times, both of them will be recorded in
		// the map
		private TreeMap<Integer, ArrayList<String>> tmap = null;
		private ArrayList<String> words = null;
		private Text frequencyandword = new Text();
		private NullWritable nwkey = NullWritable.get();

		public void setup(Context context) throws IOException,
				InterruptedException {
			// initialize tmap for current map
			tmap = new TreeMap<Integer, ArrayList<String>>();
		}

		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {
			// get word and its frequency
			String word = key.toString();
			Integer frequency = Integer.valueOf(value.toString());

			// if the frequency is in tmap, the add the word into its Arraylist
			if (tmap.containsKey(frequency)) {
				words = tmap.get(frequency);
				words.add(word);
			}
			// otherwise, insert the word into a new ArrayList(wlist)
			// and put <frequency, wlist> into tmap
			else {
				words = new ArrayList<String>();
				words.add(word);
				tmap.put(frequency, words);
			}
			// if tmap contains more than 20 pairs, remove the pair with the
			// smallest frequency
			if (tmap.size() > 20) {
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
			// combination of frequency and word (e.g. "frequency:value")
			for (Integer frequency : tmap.keySet()) {
				words = tmap.get(frequency);
				for (String word : words) {
					frequencyandword
							.set(String.valueOf(frequency) + ":" + word);
					context.write(nwkey, frequencyandword);
				}
			}

		}
	}

	// Get Global Top K
	public static class TopKReducer extends
			Reducer<NullWritable, Text, IntWritable, Text> {

		private TreeMap<Integer, ArrayList<String>> tmap = null;

		public void setup(Context context) throws IOException,
				InterruptedException {
			tmap = new TreeMap<Integer, ArrayList<String>>();

		}

		public void reduce(NullWritable key, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {

			Integer frequency = 0;
			String word = null;
			String[] frequencyandword = new String[2];
			ArrayList<String> words = null;
			for (Text val : values) {
				// parse values into <frequency, word> pairs and put them into
				// TreeMap(tmap) one by one.
				frequencyandword = val.toString().split(":");
				frequency = Integer.valueOf(frequencyandword[0]);
				word = frequencyandword[1];

				if (tmap.containsKey(frequency)) {
					words = tmap.get(frequency);
					words.add(word);
				} else {
					words = new ArrayList<String>();
					words.add(word);
					tmap.put(frequency, words);
				}
				// Whenever there are more than 20 pairs
				// in tmap, remove one pair with the smallest frequency. The
				// operations here are similar to those in the map function.
				if (tmap.size() > 20) {
					tmap.remove(tmap.firstKey());
				}
			}

		}

		public void cleanup(Context context) throws IOException,
				InterruptedException {
			// output 20 pairs in TreeMap(tmap) reversely

			// copy treemap into navigablemap in a reverse order of keys
			NavigableMap<Integer, ArrayList<String>> reverseTreeMap = tmap
					.descendingMap();
			int count = 0;
			Text value = new Text();
			IntWritable key = new IntWritable();
			for (Map.Entry<Integer, ArrayList<String>> entry : reverseTreeMap
					.entrySet()) {
				int frequency = entry.getKey();
				ArrayList<String> words = entry.getValue();
				for (String word : words) {
					key.set(frequency);
					value.set(word);
					context.write(key, value);
					count++;
				}
				if (count >= 20)
					break;

			}
		}
	}

	public static void main(String[] args) throws IOException,
			InterruptedException, ClassNotFoundException {
		// TODO Auto-generated method stub
		long start = System.currentTimeMillis();
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args)
				.getRemainingArgs();
		if (otherArgs.length != 4) {
			System.err
					.println("Usage: TopKCommonWords <input1> <input2> <output1> "
							+ "<output2>");
			System.exit(2);
		}

		Job job1 = new Job(conf, "Count Commond Words");
		job1.setJarByClass(TopKCommonWordsO.class);
		MultipleInputs.addInputPath(job1, new Path(otherArgs[0]),
				TextInputFormat.class, TokenizerWCMapper1.class);
		MultipleInputs.addInputPath(job1, new Path(otherArgs[1]),
				TextInputFormat.class, TokenizerWCMapper2.class);
		job1.setMapOutputKeyClass(Text.class);
		job1.setMapOutputValueClass(Text.class);
		job1.setCombinerClass(WCCombiner.class);
		job1.setReducerClass(CommonWordsReducer.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(IntWritable.class);
		job1.setNumReduceTasks(3);
		FileOutputFormat.setOutputPath(job1, new Path(otherArgs[2]));
		job1.waitForCompletion(true);

		Job job2 = new Job(conf, "Get Top K Common Words");
		job2.setJarByClass(TopKCommonWordsO.class);
		job2.setInputFormatClass(KeyValueTextInputFormat.class);
		FileInputFormat.addInputPath(job2, new Path(otherArgs[2]));
		job2.setMapperClass(TopKMapper.class);
		job2.setMapOutputKeyClass(NullWritable.class);
		job2.setMapOutputValueClass(Text.class);

		job2.setReducerClass(TopKReducer.class);
		job2.setOutputKeyClass(IntWritable.class);
		job2.setOutputValueClass(Text.class);

		job2.setNumReduceTasks(1);

		FileOutputFormat.setOutputPath(job2, new Path(otherArgs[3]));
		job2.waitForCompletion(true);

		long end = System.currentTimeMillis();
		System.out.println("Total time spent: " + (end - start));

	}

}