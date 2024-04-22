import { NextApiRequest, NextApiResponse } from 'next';
import { getSession } from 'next-auth/react'; 
import { sentimentAnalysisFunction } from '../../ml/sentimentAnalysis'; 

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
    const session = await getSession({ req });

    if (!session) {
        res.status(401).json({ error: 'Unauthorized' });
        return;
    }

    if (req.method !== 'POST') {
        res.status(405).json({ error: 'Method Not Allowed' });
        return;
    }

    const userInput: string = req.body.text;
    try {
        const sentimentResult: string = await sentimentAnalysisFunction(userInput); 
        res.status(200).json({ sentiment: sentimentResult }); 
    } catch (error) {
        res.status(500).json({ error: 'Internal Server Error' });
    }
}
