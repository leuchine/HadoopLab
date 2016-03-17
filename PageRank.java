package hadooplab1;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class PageRank {
	// record the number of nodes;
	public static int count = 0;

	// Compute one iteration of PageRank.

	public static class Mapper1 extends Mapper<Object, Text, Text, Text> {

		private Text k = new Text();
		private Text v = new Text();

		public void map(Object key, Text value, Context context)

		throws IOException, InterruptedException {
			// parse an input line into page, pagerank, outgoing links
			String[] list = value.toString().split("\\s");

			double pagerankScore = Double.parseDouble(list[0]);
			// We need to output both graph structure and the credit sent to
			// links

			// Graph structure: output a pair of (page, “EDGE:”+outgoing links)
			String graph = "EDGE:";
			for (int i = 1; i < list.length; i++) {
				if (i != list.length - 1) {
					graph += list[i] + " ";
				} else
					graph += list[i];
			}

			k.set(key.toString().trim());
			v.set(graph.trim());
			context.write(k, v);

			// Credit: for each outgoing link, output a pair (link,
			// pagerank/number of outgoing links)
			for (int i = 1; i < list.length; i++) {
				k.set(list[i].trim());
				v.set(Double.toString(pagerankScore / (list.length - 1)));
				context.write(k, v);

			}
		}
	}

	public static class Reducer1 extends Reducer<Text, Text, Text, Text> {
		private Text v = new Text();

		public void reduce(Text key, Iterable<Text> value, Context context)

		throws IOException, InterruptedException {
			// analyze values, if the value starts with “EDGE:”, then the phrase
			// after “EDGE:” are outgoing links

			// sum up the values that do not start with “EDGE:” into a variable
			// S
			if (key.toString().equals("1"))
				System.out.println("here");
			double S = 0;
			String graph = null;
			ArrayList<String> a = new ArrayList<String>();
			for (Text val : value) {
				a.add(val.toString());
			}

			for (int i = 0; i < a.size(); i++) {
				String val = a.get(i);
				if (val.toString().startsWith("EDGE:")) {
					if (val.toString().equals("EDGE:"))
						graph = "";
					else
						graph = val.toString().substring(5);
				} else {
					S += Double.parseDouble(val.toString());
				}

			}

			// compute new pagerank as 0.15/N+0.85*S (N is the total number of
			// nodes)
			double newPageRank = 0.15 / PageRank.count + 0.85 * S;
			// output (key, newpagerank + outgoing links)

			if (graph.equals("")) {
				v.set(Double.toString(newPageRank));
			} else
				v.set(Double.toString(newPageRank) + " " + graph);
			context.write(key, v);
		}

	}

	// Count the number of nodes and attach a pagerank score 1
	public static void preprocessing(String filename) {
		Scanner scan = null;
		try {
			scan = new Scanner(new File(filename));
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		PrintWriter pw = null;
		try {
			pw = new PrintWriter("/search/0");
		} catch (FileNotFoundException e) {
		}
		count = 0;
		while (scan.hasNextLine()) {
			String line = scan.nextLine();
			String[] elements = line.split("\t");
			count++;
			if (elements.length == 1) {
				pw.println(elements[0].trim() + "\t" + 1);
				continue;
			}
			for (int i = 0; i < elements.length; i++) {
				if (i == 0) {
					pw.print(elements[0].trim() + "\t" + 1 + " ");
				} else if (i != elements.length - 1) {
					pw.print(elements[i].trim() + " ");
				} else {
					pw.println(elements[i].trim());
				}
			}

		}
		pw.flush();
	}

	public static void main(String[] args) throws IOException,
			InterruptedException, ClassNotFoundException {
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args)
				.getRemainingArgs();
		if (otherArgs.length != 1) {
			System.err.println("Usage: PageRank <input1>");
			System.exit(2);
		}
		preprocessing(otherArgs[0]);
		for (int i = 0; i < 10; i++) {
			// create a new job, set job configurations and run the job
			Job job = new Job(conf, "pagerank" + i);

			job.setJarByClass(PageRank.class);
			job.setInputFormatClass(KeyValueTextInputFormat.class);
			job.setMapperClass(Mapper1.class);
			job.setMapOutputKeyClass(Text.class);
			job.setMapOutputValueClass(Text.class);
			job.setReducerClass(Reducer1.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(Text.class);
			FileInputFormat.addInputPath(job,
					new Path("/search/" + Integer.toString(i)));
			FileOutputFormat.setOutputPath(job,
					new Path("/search/" + Integer.toString(i + 1)));
			job.waitForCompletion(true);
		}

	}
}